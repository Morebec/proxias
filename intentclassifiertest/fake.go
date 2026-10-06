package intentclassifiertest

import (
	"context"

	"github.com/morebec/proxias"
)

// Fixed always answers the same Intent and error, and counts its calls.
type Fixed struct {
	Intent proxias.Intent
	Err    error
	Calls  int
}

func (f *Fixed) Classify(context.Context, proxias.Request, []proxias.Candidate) (proxias.Intent, error) {
	f.Calls++
	return f.Intent, f.Err
}
