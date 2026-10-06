package proxias

// Outcome is how a request was answered: by Workflows alone, or by the Fallback.
type Outcome struct {
	// Results are the Results of the Workflows run, in order. Empty when none ran.
	Results []Result
	// Completion is the Fallback's answer, set when the Fallback answered (after the Results, if any).
	Completion any
	// FellBack is true when the Fallback answered.
	FellBack bool
}
