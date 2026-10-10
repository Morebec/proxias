package proxias

import (
	"encoding/json"
	"reflect"
	"strings"
)

// This file keeps the fields of a request that the model does not name. A proxy that forwards a request must
// not drop what it does not understand, so each request-side type keeps the other fields of its JSON object in
// Extra, and writes them back when it is marshalled. The typed fields are the source of truth: a middleware that
// changes them changes what is sent, and the extra fields never override a typed one.

// unmarshalPlain reads data into plain, a pointer to a type without these methods, and returns the fields of
// data that plain does not name.
func unmarshalPlain(data []byte, plain any) (map[string]json.RawMessage, error) {
	if err := json.Unmarshal(data, plain); err != nil {
		return nil, err
	}

	var all map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return nil, err
	}

	named := jsonNames(reflect.TypeOf(plain).Elem())
	for name := range all {
		if named[name] {
			delete(all, name)
		}
	}
	if len(all) == 0 {
		return nil, nil
	}
	return all, nil
}

// marshalPlain writes plain, a type without these methods, and the extra fields that its names do not hide.
func marshalPlain(plain any, extra map[string]json.RawMessage) ([]byte, error) {
	data, err := json.Marshal(plain)
	if err != nil || len(extra) == 0 {
		return data, err
	}

	var all map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return nil, err
	}
	for name, value := range extra {
		if _, ok := all[name]; !ok {
			all[name] = value
		}
	}
	return json.Marshal(all)
}

// jsonNames are the JSON names of the exported fields of the struct type t.
func jsonNames(t reflect.Type) map[string]bool {
	names := make(map[string]bool, t.NumField())
	for i := range t.NumField() {
		field := t.Field(i)
		tag := field.Tag.Get("json")
		if !field.IsExported() || tag == "-" {
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if name == "" {
			name = field.Name
		}
		names[name] = true
	}
	return names
}
