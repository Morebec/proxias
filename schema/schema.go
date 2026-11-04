// Package schema builds the JSON Schema of the parameters of an action (an
// ActionDescriptor's InputSchema), one property per call:
//
//	schema.Object().
//		String("title", "What is to be done, in a few words.").Required().
//		Strings("checklist", "The steps to check as they are done.").
//		Integer("minutes", "How many minutes to wait.").Min(1).Max(1440).
//		Map()
//
// A modifier (Required, Min, Max) applies to the property added last.
package schema

// ObjectSchema is the JSON Schema of an object, built one property at a time.
type ObjectSchema struct {
	properties map[string]any
	required   []string
	// last is the name of the property added last, which the modifiers apply
	// to.
	last string
}

// Object returns the schema of an object with no property yet.
func Object() *ObjectSchema {
	return &ObjectSchema{properties: map[string]any{}}
}

// String adds a property of text.
func (s *ObjectSchema) String(name, description string) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type": "string",
	})
}

// Strings adds a property that is a list of texts.
func (s *ObjectSchema) Strings(name, description string) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "string",
		},
	})
}

// Enum adds a property of text that is one of the values.
func (s *ObjectSchema) Enum(name, description string, values ...string) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type": "string",
		"enum": values,
	})
}

// Integer adds a property that is a whole number.
func (s *ObjectSchema) Integer(name, description string) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type": "integer",
	})
}

// Bool adds a property that is true or false.
func (s *ObjectSchema) Bool(name, description string) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type": "boolean",
	})
}

// AnyObject adds a property that is an object of any properties.
func (s *ObjectSchema) AnyObject(name, description string) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type": "object",
	})
}

// Objects adds a property that is a list of objects of the schema of.
func (s *ObjectSchema) Objects(name, description string, of *ObjectSchema) *ObjectSchema {
	return s.add(name, description, map[string]any{
		"type":  "array",
		"items": of.Map(),
	})
}

// Required makes the property added last required.
func (s *ObjectSchema) Required() *ObjectSchema {
	s.required = append(s.required, s.last)
	return s
}

// Min is the least value of the property added last.
func (s *ObjectSchema) Min(value int) *ObjectSchema {
	s.properties[s.last].(map[string]any)["minimum"] = value
	return s
}

// Max is the greatest value of the property added last.
func (s *ObjectSchema) Max(value int) *ObjectSchema {
	s.properties[s.last].(map[string]any)["maximum"] = value
	return s
}

// Map is the schema, as an ActionDescriptor's InputSchema: its required
// properties are left out while it has none.
func (s *ObjectSchema) Map() map[string]any {
	schema := map[string]any{
		"type":       "object",
		"properties": s.properties,
	}
	if len(s.required) > 0 { // a vendor refuses a null list
		schema["required"] = s.required
	}
	return schema
}

// add adds the property, with its description when it has one.
func (s *ObjectSchema) add(name, description string, property map[string]any) *ObjectSchema {
	if description != "" {
		property["description"] = description
	}
	s.properties[name] = property
	s.last = name
	return s
}
