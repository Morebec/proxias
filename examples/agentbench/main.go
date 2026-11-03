// Command agentbench compares the strategies on the scenarios of the agent examples: it runs each scenario on each
// strategy, and tells, per scenario and strategy, how often the task succeeded, the mean and spread of the calls to
// Claude, their tokens and cost, and the time a run took: in all, to its first action request or reply, per call to
// Claude, per turn of the scenario's agents, and per stage of the gated strategies (summed over a run; a delegated
// agent's stages count in the run, inside the execute stage of the agent that asked it).
//
// Runs go -parallel at a time, each with its own agents, sessions and counters. A call is never retried: a rate
// limit (429) fails its run, and is counted. Concurrent runs may raise the latency of calls.
//
//	go run ./examples/agentbench                         # one run of each on scripts: checks the bench, costs nothing
//	go run ./examples/agentbench -runs 10 -parallel 6    # ten runs of each on Claude (billed)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias/agentsdk"
	agentanthropic "github.com/morebec/proxias/agentsdk/anthropic"
	"github.com/morebec/proxias/examples/agentdemo"
)

var (
	runs       = flag.Int("runs", 0, "runs of each scenario on each strategy, on Claude ("+agentdemo.ClaudeModel+"); 0 runs each once on scripts")
	parallel   = flag.Int("parallel", 1, "runs at a time")
	suite      = flag.String("suite", "basic", "the scenarios: basic (the examples', each agent with its own action), or catalog (chat and unattended, the agent with a hundred actions)")
	strategies = flag.String("strategies", "", "the strategies to compare, comma separated; by default all of them for basic, "+agentdemo.StrategyReAct+" and "+agentdemo.StrategyGated+" for catalog")
	scenarios  = flag.String("scenarios", "", "the scenarios to run, comma separated; all when empty")
	verbose    = flag.Bool("v", false, "print the events of each run")
)

// scenario is a task an agent is given, and how to tell it succeeded.
type scenario struct {
	name string
	// run runs the task, its agents on the strategy of the run, and tells whether it succeeded, and why not.
	run func(ctx context.Context, r *run) (bool, string, error)
}

// Prices, in dollars per million tokens: the list prices of Claude Haiku 5.5; cache writes and reads are 1.25 and
// 0.1 times the input price.
const (
	inputPrice      = 0.10
	outputPrice     = 0.50
	cacheWritePrice = 0.125
	cacheReadPrice  = 0.01
)

// measure is what one run cost, and how long it took (in ms).
type measure struct {
	ok bool
	// why is why it failed, "" when it succeeded.
	why string
	// rateLimited tells whether it failed on a rate limit (429).
	rateLimited                                         bool
	calls, input, output, cacheWrite, cacheRead, millis float64
	// perCall is the mean latency of the run's calls to Claude.
	perCall float64
	// firstEvent is when the first action request or reply of an agent came, since the run began.
	firstEvent float64
	// turns are the turns of the run's agents, and perTurn their mean duration.
	turns, perTurn float64
	// stages is the time spent in each stage of the gated strategies, summed over the run.
	stages map[agentsdk.GatedStage]float64
	// events are the events of the run, printed with -v.
	events []string
}

// dollars is what the run cost, at the list prices.
func (m measure) dollars() float64 {
	return (m.input*inputPrice + m.output*outputPrice + m.cacheWrite*cacheWritePrice + m.cacheRead*cacheReadPrice) / 1e6
}

// stages are the stages of the gated strategy, in the order of the table.
var stages = []agentsdk.GatedStage{agentsdk.GatedClassify, agentsdk.GatedExecute, agentsdk.GatedSynthesize, agentsdk.GatedFallback}

// timing records the times and the events of a run as they come.
type timing struct {
	mutex      sync.Mutex
	began      time.Time
	firstEvent time.Duration
	turns      []time.Duration
	stages     map[agentsdk.GatedStage]time.Duration
	events     []string
}

func (t *timing) observe(e agentsdk.Event) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	useful := e.Type == agentsdk.EventTypeActionRequest ||
		(e.Type == agentsdk.EventTypeMessage && len(e.Payload) > 0)
	if t.firstEvent == 0 && e.Source.Type == agentsdk.EventSourceTypeAgent && useful {
		t.firstEvent = time.Since(t.began)
	}
	if *verbose {
		t.events = append(t.events, agentdemo.Line(e))
	}
}

func (t *timing) cognition(d time.Duration) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	t.turns = append(t.turns, d)
}

func (t *timing) stage(stage agentsdk.GatedStage, d time.Duration) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	t.stages[stage] += d
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// cell is a scenario on a strategy.
type cell struct {
	scenario scenario
	kind     string
}

func main() {
	flag.Parse()
	onClaude := *runs > 0
	n := max(*runs, 1)

	kinds := *strategies
	if kinds == "" {
		kinds = strings.Join(agentdemo.Strategies, ",")
		if *suite == "catalog" {
			kinds = agentdemo.StrategyReAct + "," + agentdemo.StrategyGated
		}
	}
	suites := map[string]func() []scenario{"basic": all, "catalog": catalogScenarios}
	of, ok := suites[*suite]
	if !ok {
		fmt.Fprintf(os.Stderr, "-suite %q: it is basic or catalog\n", *suite)
		os.Exit(2)
	}
	if *suite == "catalog" {
		descs, _ := catalogActions(&calls{}).Descriptors(context.Background(), agentsdk.Session{})
		data, _ := json.Marshal(descs)
		fmt.Printf("catalog: %d actions, %d bytes of descriptors with their schemas (about %d tokens)\n\n", len(descs), len(data), len(data)/4)
	}
	var cells []cell
	for _, sc := range of() {
		if *scenarios != "" && !slices.Contains(strings.Split(*scenarios, ","), sc.name) {
			continue
		}
		for _, kind := range strings.Split(kinds, ",") {
			cells = append(cells, cell{scenario: sc, kind: kind})
		}
	}

	measures := make([][]measure, len(cells))
	for i := range measures {
		measures[i] = make([]measure, n)
	}
	var (
		wait  sync.WaitGroup
		print sync.Mutex
		slots = make(chan struct{}, max(*parallel, 1))
	)
	for c := range cells {
		for r := range n {
			slots <- struct{}{}
			wait.Go(func() {
				defer func() { <-slots }()
				m := runOnce(cells[c], onClaude)
				measures[c][r] = m
				if *verbose {
					print.Lock()
					fmt.Printf("== %s on %s, run %d\n%s\n", cells[c].scenario.name, cells[c].kind, r+1, strings.Join(m.events, "\n"))
					if !m.ok {
						fmt.Printf("failed: %s\n", m.why)
					}
					print.Unlock()
				}
			})
		}
	}
	wait.Wait()
	report(cells, measures)
}

// runOnce runs the scenario of the cell once, on its strategy, in an env of its own.
func runOnce(c cell, onClaude bool) measure {
	t := &timing{stages: map[agentsdk.GatedStage]time.Duration{}}
	env := agentdemo.Env{Counter: &agentanthropic.UsageCounter{}, StageTimer: t.stage}
	t.began = time.Now()
	ok, why, err := c.scenario.run(context.Background(), &run{env: env, kind: c.kind, onClaude: onClaude, t: t})
	usage := env.Counter.Usage()
	var apiErr *anthropicsdk.Error
	rateLimited := errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests
	if err != nil {
		why = err.Error()
	}

	t.mutex.Lock()
	defer t.mutex.Unlock()
	m := measure{
		ok:          ok && err == nil,
		why:         why,
		rateLimited: rateLimited,
		calls:       float64(usage.Calls),
		input:       float64(usage.InputTokens),
		output:      float64(usage.OutputTokens),
		cacheWrite:  float64(usage.CacheCreationInputTokens),
		cacheRead:   float64(usage.CacheReadInputTokens),
		millis:      float64(time.Since(t.began).Milliseconds()),
		firstEvent:  ms(t.firstEvent),
		turns:       float64(len(t.turns)),
		stages:      map[agentsdk.GatedStage]float64{},
		events:      t.events,
	}
	if usage.Calls > 0 {
		m.perCall = ms(usage.CallDuration) / float64(usage.Calls)
	}
	var took time.Duration
	for _, d := range t.turns {
		took += d
	}
	if len(t.turns) > 0 {
		m.perTurn = ms(took) / float64(len(t.turns))
	}
	for stage, d := range t.stages {
		m.stages[stage] = ms(d)
	}
	return m
}

// report prints the tables of the cells, their failures, and the total.
func report(cells []cell, measures [][]measure) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(w, "scenario\tstrategy\truns\tsuccess\tcalls\tinput tokens\toutput tokens\tcache write\tcache read\tms\t$ per run\t")
	times := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(times, "scenario\tstrategy\tms\tfirst event ms\tms per call\tturns\tms per turn\tclassify ms\texecute ms\tsynthesize ms\tfallback ms\t")
	var (
		total       measure
		dollars     float64
		failures    = map[string]int{}
		rateLimited int
	)
	for c, cell := range cells {
		ms := measures[c]
		succeeded := 0
		for _, m := range ms {
			if m.ok {
				succeeded++
			} else {
				failures[fmt.Sprintf("%s on %s: %s", cell.scenario.name, cell.kind, m.why)]++
			}
			if m.rateLimited {
				rateLimited++
			}
			total.calls += m.calls
			total.input += m.input
			total.output += m.output
			total.cacheWrite += m.cacheWrite
			total.cacheRead += m.cacheRead
			dollars += m.dollars()
		}
		fmt.Fprintf(w, "%s\t%s\t%d\t%d/%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n", cell.scenario.name, cell.kind, len(ms), succeeded, len(ms),
			stat(ms, func(m measure) float64 { return m.calls }),
			stat(ms, func(m measure) float64 { return m.input }),
			stat(ms, func(m measure) float64 { return m.output }),
			stat(ms, func(m measure) float64 { return m.cacheWrite }),
			stat(ms, func(m measure) float64 { return m.cacheRead }),
			stat(ms, func(m measure) float64 { return m.millis }),
			dollarStat(ms))
		row := make([]string, 0, 7+len(stages))
		row = append(row, cell.scenario.name, cell.kind,
			stat(ms, func(m measure) float64 { return m.millis }),
			stat(ms, func(m measure) float64 { return m.firstEvent }),
			stat(ms, func(m measure) float64 { return m.perCall }),
			stat(ms, func(m measure) float64 { return m.turns }),
			stat(ms, func(m measure) float64 { return m.perTurn }),
		)
		for _, stage := range stages {
			row = append(row, stat(ms, func(m measure) float64 { return m.stages[stage] }))
		}
		fmt.Fprintln(times, strings.Join(row, "\t")+"\t")
	}
	_ = w.Flush()
	fmt.Println()
	_ = times.Flush()
	fmt.Printf("\ntotal: %.0f calls, %.0f input, %.0f output, %.0f cache write, %.0f cache read tokens: $%.4f at list prices; %d runs failed on a rate limit (429)\n",
		total.calls, total.input, total.output, total.cacheWrite, total.cacheRead, dollars, rateLimited)
	keys := make([]string, 0, len(failures))
	for k := range failures {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		fmt.Printf("%4d failed: %s\n", failures[k], k)
	}
}

// dollarStat is the mean ± the standard deviation of what the measures cost.
func dollarStat(measures []measure) string {
	return spread(measures, measure.dollars, "%.5f ± %.5f")
}

// stat is the mean ± the standard deviation of the value of the measures.
func stat(measures []measure, value func(measure) float64) string {
	return spread(measures, value, "%.0f ± %.0f")
}

// spread formats the mean and the standard deviation of the value of the measures.
func spread(measures []measure, value func(measure) float64, format string) string {
	var sum float64
	for _, m := range measures {
		sum += value(m)
	}
	mean := sum / float64(len(measures))
	var squares float64
	for _, m := range measures {
		squares += (value(m) - mean) * (value(m) - mean)
	}
	return fmt.Sprintf(format, mean, math.Sqrt(squares/float64(len(measures))))
}
