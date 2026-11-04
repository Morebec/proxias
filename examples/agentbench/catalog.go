package main

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/morebec/proxias"
)

// tool is an action of the catalog: its ID, what it does, and its parameters,
// each "name:type:description", a
// leading "*" for a required one (type string, number or strings).
type tool struct {
	id, description string
	params          []string
}

// catalog is an operations assistant's hundred actions, over ten domains, some
// of them alike on purpose (close, resolve and archive a ticket; an invoice and
// a quote; send, draft and reply to an email...).
var catalog = []tool{
	// CRM
	{
		"crm_find_customer",
		"Finds a customer by name, and gives its ID, name and main contact's " +
			"email.",
		[]string{"*name:string:the customer's name, or part of it"},
	},
	{
		"crm_get_customer",
		"Gives the full record of a customer, by ID.",
		[]string{"*customer_id:string:the customer's ID"},
	},
	{
		"crm_create_customer",
		"Creates a customer record.",
		[]string{
			"*name:string:the customer's name",
			"*email:string:the main contact's email",
			"company:string:the company it belongs to",
		},
	},
	{
		"crm_update_customer",
		"Updates the contact details of a customer.",
		[]string{
			"*customer_id:string:the customer's ID",
			"email:string:the new email",
			"phone:string:the new phone number",
		},
	},
	{
		"crm_delete_customer",
		"Deletes a customer record, for good.",
		[]string{"*customer_id:string:the customer's ID"},
	},
	{
		"crm_list_customers",
		"Lists the customers, of a segment if one is given.",
		[]string{"segment:string:the segment (smb, enterprise...)"},
	},
	{
		"crm_add_note",
		"Adds a note to a customer's record.",
		[]string{
			"*customer_id:string:the customer's ID",
			"*note:string:the note",
		},
	},
	{
		"crm_find_contact",
		"Finds a person among the contacts of all customers, by name.",
		[]string{"*name:string:the person's name"},
	},
	{
		"crm_merge_customers",
		"Merges a duplicate customer record into another.",
		[]string{
			"*primary_id:string:the record kept",
			"*duplicate_id:string:the record merged into it",
		},
	},
	{
		"crm_assign_owner",
		"Assigns the account owner of a customer.",
		[]string{
			"*customer_id:string:the customer's ID",
			"*owner:string:the owner's name",
		},
	},
	// Billing
	{
		"billing_create_invoice",
		"Creates an invoice, a request for payment of an amount by a due date.",
		[]string{
			"*customer:string:the customer billed",
			"*amount:number:the amount",
			"*currency:string:the currency code (USD, EUR...)",
			"*due_date:string:the due date, YYYY-MM-DD",
		},
	},
	{
		"billing_create_quote",
		"Creates a quote, a price offered to a customer, valid until a date; " +
			"it is not a request for payment.",
		[]string{
			"*customer:string:the customer",
			"*amount:number:the amount",
			"*currency:string:the currency code",
			"*valid_until:string:the last day it is valid, YYYY-MM-DD",
		},
	},
	{
		"billing_send_invoice",
		"Sends an existing invoice by email.",
		[]string{
			"*invoice_id:string:the invoice's ID",
			"*email:string:where to send it",
		},
	},
	{
		"billing_void_invoice",
		"Voids an invoice that should not have been issued.",
		[]string{"*invoice_id:string:the invoice's ID", "*reason:string:why"},
	},
	{
		"billing_refund_payment",
		"Refunds all or part of a payment.",
		[]string{
			"*payment_id:string:the payment's ID",
			"*amount:number:the amount refunded",
		},
	},
	{
		"billing_record_payment",
		"Records a payment received for an invoice.",
		[]string{
			"*invoice_id:string:the invoice's ID",
			"*amount:number:the amount received",
		},
	},
	{
		"billing_list_invoices",
		"Lists the invoices, of a customer if one is given.",
		[]string{"customer:string:the customer"},
	},
	{
		"billing_get_invoice",
		"Gives an invoice, by ID.",
		[]string{"*invoice_id:string:the invoice's ID"},
	},
	{
		"billing_create_credit_note",
		"Creates a credit note against an invoice.",
		[]string{
			"*invoice_id:string:the invoice's ID",
			"*amount:number:the amount credited",
		},
	},
	{
		"billing_update_due_date",
		"Changes the due date of an invoice.",
		[]string{
			"*invoice_id:string:the invoice's ID",
			"*due_date:string:the new due date, YYYY-MM-DD",
		},
	},
	// Tickets
	{
		"tickets_create",
		"Opens a support ticket.",
		[]string{
			"*title:string:the title",
			"*description:string:what is wrong",
			"priority:string:low, normal, high or urgent",
		},
	},
	{
		"tickets_get",
		"Gives a ticket, by ID.",
		[]string{"*ticket_id:string:the ticket's ID"},
	},
	{
		"tickets_close",
		"Closes a ticket without a resolution: it was a duplicate, invalid, " +
			"or abandoned.",
		[]string{"*ticket_id:string:the ticket's ID"},
	},
	{
		"tickets_resolve",
		"Marks a ticket resolved, with a note saying how the problem was " +
			"fixed.",
		[]string{
			"*ticket_id:string:the ticket's ID",
			"*resolution_note:string:how it was fixed",
		},
	},
	{
		"tickets_archive",
		"Archives a ticket: it leaves the lists, whatever its state.",
		[]string{"*ticket_id:string:the ticket's ID"},
	},
	{
		"tickets_reopen",
		"Reopens a closed or resolved ticket.",
		[]string{"*ticket_id:string:the ticket's ID", "*reason:string:why"},
	},
	{
		"tickets_assign",
		"Assigns a ticket to someone.",
		[]string{"*ticket_id:string:the ticket's ID", "*assignee:string:who"},
	},
	{
		"tickets_comment",
		"Adds a comment to a ticket, without changing its state.",
		[]string{
			"*ticket_id:string:the ticket's ID",
			"*comment:string:the comment",
		},
	},
	{
		"tickets_escalate",
		"Escalates a ticket to a higher support level.",
		[]string{
			"*ticket_id:string:the ticket's ID",
			"*level:string:the level (l2, l3)",
		},
	},
	{
		"tickets_list",
		"Lists the tickets, of a status if one is given.",
		[]string{"status:string:open, resolved, closed"},
	},
	// Email
	{
		"email_send",
		"Sends a new email now.",
		[]string{
			"*to:string:the recipient's address",
			"*subject:string:the subject",
			"*body:string:the text",
		},
	},
	{
		"email_reply",
		"Replies to an email received.",
		[]string{
			"*message_id:string:the email answered",
			"*body:string:the text",
		},
	},
	{
		"email_forward",
		"Forwards an email received to someone.",
		[]string{
			"*message_id:string:the email forwarded",
			"*to:string:the recipient's address",
		},
	},
	{
		"email_draft",
		"Writes an email as a draft, without sending it.",
		[]string{
			"*to:string:the recipient's address",
			"*subject:string:the subject",
			"*body:string:the text",
		},
	},
	{
		"email_send_draft",
		"Sends a draft written before.",
		[]string{"*draft_id:string:the draft's ID"},
	},
	{
		"email_list_inbox",
		"Lists the emails of a folder.",
		[]string{"folder:string:the folder, the inbox by default"},
	},
	{
		"email_search",
		"Searches the emails.",
		[]string{"*query:string:what to look for"},
	},
	{
		"email_archive",
		"Archives an email.",
		[]string{"*message_id:string:the email's ID"},
	},
	{
		"email_schedule",
		"Schedules an email to be sent later.",
		[]string{
			"*to:string:the recipient's address",
			"*subject:string:the subject",
			"*body:string:the text",
			"*send_at:string:when, ISO 8601",
		},
	},
	{
		"email_delete",
		"Deletes an email.",
		[]string{"*message_id:string:the email's ID"},
	},
	// Calendar
	{
		"calendar_create_event",
		"Creates a meeting in the calendar, and invites its attendees.",
		[]string{
			"*title:string:the title",
			"*attendees:strings:the people invited",
			"*start:string:when it starts, ISO 8601",
			"*duration_minutes:number:how long it lasts",
		},
	},
	{
		"calendar_update_event",
		"Changes the time or length of a meeting.",
		[]string{
			"*event_id:string:the meeting's ID",
			"start:string:the new start",
			"duration_minutes:number:the new length",
		},
	},
	{
		"calendar_cancel_event",
		"Cancels a meeting.",
		[]string{"*event_id:string:the meeting's ID"},
	},
	{
		"calendar_list_events",
		"Lists the meetings of a day.",
		[]string{"*date:string:the day, YYYY-MM-DD"},
	},
	{
		"calendar_find_free_slot",
		"Finds a time when all the attendees are free.",
		[]string{
			"*attendees:strings:the people",
			"*duration_minutes:number:how long",
			"*date:string:the day, YYYY-MM-DD",
		},
	},
	{
		"calendar_invite",
		"Invites someone to an existing meeting.",
		[]string{"*event_id:string:the meeting's ID", "*attendee:string:who"},
	},
	{
		"calendar_create_reminder",
		"Creates a reminder for oneself.",
		[]string{"*text:string:what to remember", "*at:string:when, ISO 8601"},
	},
	{
		"calendar_decline_event",
		"Declines a meeting one is invited to.",
		[]string{"*event_id:string:the meeting's ID"},
	},
	{
		"calendar_reschedule_event",
		"Moves a meeting to another time.",
		[]string{
			"*event_id:string:the meeting's ID",
			"*start:string:the new start, ISO 8601",
		},
	},
	{
		"calendar_book_room",
		"Books a meeting room.",
		[]string{
			"*room:string:the room",
			"*start:string:when, ISO 8601",
			"*duration_minutes:number:how long",
		},
	},
	// Inventory
	{
		"inventory_check_stock",
		"Gives the stock of a product.",
		[]string{"*sku:string:the product's SKU"},
	},
	{
		"inventory_adjust_stock",
		"Adjusts the stock of a product by a quantity, after a count.",
		[]string{
			"*sku:string:the product's SKU",
			"*delta:number:the change, negative to remove",
		},
	},
	{
		"inventory_reserve_stock",
		"Reserves stock for an order.",
		[]string{
			"*sku:string:the product's SKU",
			"*quantity:number:how many",
			"*order_id:string:the order",
		},
	},
	{
		"inventory_release_reservation",
		"Releases a reservation of stock.",
		[]string{"*reservation_id:string:the reservation's ID"},
	},
	{
		"inventory_create_purchase_order",
		"Orders stock from a supplier.",
		[]string{
			"*sku:string:the product's SKU",
			"*quantity:number:how many",
			"*supplier:string:the supplier",
		},
	},
	{
		"inventory_receive_shipment",
		"Records the stock received for a purchase order.",
		[]string{"*purchase_order_id:string:the purchase order's ID"},
	},
	{
		"inventory_list_low_stock",
		"Lists the products whose stock is low.",
		[]string{"threshold:number:the threshold"},
	},
	{
		"inventory_move_stock",
		"Moves stock from a warehouse to another.",
		[]string{
			"*sku:string:the product's SKU",
			"*from_warehouse:string:from",
			"*to_warehouse:string:to",
			"*quantity:number:how many",
		},
	},
	{
		"inventory_get_product",
		"Gives a product, by SKU.",
		[]string{"*sku:string:the product's SKU"},
	},
	{
		"inventory_update_price",
		"Changes the price of a product.",
		[]string{
			"*sku:string:the product's SKU",
			"*price:number:the new price",
		},
	},
	// Orders
	{
		"orders_create",
		"Creates an order for a customer.",
		[]string{
			"*customer:string:the customer",
			"*sku:string:the product's SKU",
			"*quantity:number:how many",
		},
	},
	{
		"orders_get",
		"Gives an order, by ID.",
		[]string{"*order_id:string:the order's ID"},
	},
	{
		"orders_cancel",
		"Cancels an order.",
		[]string{"*order_id:string:the order's ID", "*reason:string:why"},
	},
	{
		"orders_ship",
		"Ships an order with a carrier.",
		[]string{
			"*order_id:string:the order's ID",
			"*carrier:string:the carrier",
		},
	},
	{
		"orders_track",
		"Gives where the shipment of an order is.",
		[]string{"*order_id:string:the order's ID"},
	},
	{
		"orders_return",
		"Opens the return of an order.",
		[]string{"*order_id:string:the order's ID", "*reason:string:why"},
	},
	{
		"orders_update_address",
		"Changes the shipping address of an order.",
		[]string{
			"*order_id:string:the order's ID",
			"*address:string:the new address",
		},
	},
	{
		"orders_list",
		"Lists the orders, of a customer if one is given.",
		[]string{"customer:string:the customer"},
	},
	{
		"orders_mark_delivered",
		"Marks an order delivered.",
		[]string{"*order_id:string:the order's ID"},
	},
	{
		"orders_split",
		"Splits a product out of an order into an order of its own.",
		[]string{
			"*order_id:string:the order's ID",
			"*sku:string:the product split out",
		},
	},
	// HR
	{
		"hr_create_employee",
		"Creates an employee's record.",
		[]string{
			"*name:string:the name",
			"*role:string:the role",
			"*start_date:string:the first day, YYYY-MM-DD",
		},
	},
	{
		"hr_get_employee",
		"Gives an employee's record, by ID.",
		[]string{"*employee_id:string:the employee's ID"},
	},
	{
		"hr_request_time_off",
		"Requests time off for an employee.",
		[]string{
			"*employee_id:string:the employee's ID",
			"*from:string:the first day",
			"*to:string:the last day",
		},
	},
	{
		"hr_approve_time_off",
		"Approves a request of time off.",
		[]string{"*request_id:string:the request's ID"},
	},
	{
		"hr_reject_time_off",
		"Rejects a request of time off.",
		[]string{"*request_id:string:the request's ID", "*reason:string:why"},
	},
	{
		"hr_update_role",
		"Changes an employee's role.",
		[]string{
			"*employee_id:string:the employee's ID",
			"*role:string:the new role",
		},
	},
	{
		"hr_offboard_employee",
		"Starts the offboarding of an employee who leaves.",
		[]string{
			"*employee_id:string:the employee's ID",
			"*last_day:string:the last day",
		},
	},
	{
		"hr_list_employees",
		"Lists the employees, of a team if one is given.",
		[]string{"team:string:the team"},
	},
	{
		"hr_schedule_review",
		"Schedules an employee's performance review.",
		[]string{
			"*employee_id:string:the employee's ID",
			"*date:string:the day",
		},
	},
	{
		"hr_record_expense",
		"Records an expense of an employee, to be reimbursed.",
		[]string{
			"*employee_id:string:the employee's ID",
			"*amount:number:the amount",
			"*description:string:what it was",
		},
	},
	// DevOps
	{
		"devops_restart_service",
		"Restarts a service.",
		[]string{
			"*service:string:the service",
			"*environment:string:production, staging",
		},
	},
	{
		"devops_deploy_service",
		"Deploys a version of a service.",
		[]string{
			"*service:string:the service",
			"*version:string:the version",
			"*environment:string:production, staging",
		},
	},
	{
		"devops_rollback_service",
		"Rolls a service back to its previous version.",
		[]string{
			"*service:string:the service",
			"*environment:string:production, staging",
		},
	},
	{
		"devops_scale_service",
		"Changes the number of replicas of a service.",
		[]string{
			"*service:string:the service",
			"*replicas:number:how many",
			"*environment:string:production, staging",
		},
	},
	{
		"devops_get_service_status",
		"Gives the health of a service.",
		[]string{
			"*service:string:the service",
			"*environment:string:production, staging",
		},
	},
	{
		"devops_page_oncall",
		"Pages the engineer on call.",
		[]string{"*reason:string:why, in one sentence"},
	},
	{
		"devops_silence_alert",
		"Silences an alert for a while.",
		[]string{
			"*alert_id:string:the alert's ID",
			"*minutes:number:for how long",
		},
	},
	{
		"devops_acknowledge_alert",
		"Acknowledges an alert.",
		[]string{"*alert_id:string:the alert's ID"},
	},
	{
		"devops_run_backup",
		"Runs a backup of a database.",
		[]string{"*database:string:the database"},
	},
	{
		"devops_get_logs",
		"Gives the recent logs of a service.",
		[]string{
			"*service:string:the service",
			"*since:string:since when, ISO 8601",
		},
	},
	// Documents
	{
		"docs_create",
		"Creates a document.",
		[]string{"*title:string:the title", "*content:string:the text"},
	},
	{
		"docs_update",
		"Replaces the content of a document.",
		[]string{
			"*document_id:string:the document's ID",
			"*content:string:the new text",
		},
	},
	{
		"docs_share",
		"Shares a document with someone.",
		[]string{
			"*document_id:string:the document's ID",
			"*email:string:with whom",
		},
	},
	{
		"docs_search",
		"Searches the documents.",
		[]string{"*query:string:what to look for"},
	},
	{
		"docs_delete",
		"Deletes a document.",
		[]string{"*document_id:string:the document's ID"},
	},
	{
		"docs_get",
		"Gives a document, by ID.",
		[]string{"*document_id:string:the document's ID"},
	},
	{
		"docs_export",
		"Exports a document to a file format.",
		[]string{
			"*document_id:string:the document's ID",
			"*format:string:pdf, docx, md",
		},
	},
	{
		"docs_comment",
		"Comments on a document.",
		[]string{
			"*document_id:string:the document's ID",
			"*comment:string:the comment",
		},
	},
	{
		"docs_move",
		"Moves a document to a folder.",
		[]string{
			"*document_id:string:the document's ID",
			"*folder:string:the folder",
		},
	},
	{
		"docs_publish",
		"Publishes a document to the company's wiki.",
		[]string{"*document_id:string:the document's ID"},
	},
}

// call is a call of an action of the catalog, as it was made.
type call struct {
	action string
	params map[string]any
	// rejected tells that the action refused it: a required parameter was
	// missing.
	rejected bool
}

// calls records the calls of the actions of the catalog.
type calls struct {
	mutex sync.Mutex
	made  []call
}

func (c *calls) record(action string, params map[string]any, rejected bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.made = append(
		c.made,
		call{action: action, params: params, rejected: rejected},
	)
}

func (c *calls) all() []call {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return slices.Clone(c.made)
}

// catalogActions are the actions of the catalog, recorded in c: an action
// refuses a call that lacks a required
// parameter; crm_find_customer knows Globex; any other answers that it is done.
func catalogActions(c *calls) proxias.ActionExecutor {
	executor := proxias.SimpleActionExecutor{
		Actions: map[string]proxias.SimpleAction{},
	}
	for _, t := range catalog {
		schema, required := schemaOf(t.params)
		executor.Actions[t.id] = proxias.SimpleAction{
			ActionDescriptor: proxias.ActionDescriptor{
				ID:          t.id,
				Name:        t.id,
				Description: t.description,
				InputSchema: schema,
			},
			RunFn: func(
				_ context.Context,
				_ proxias.Session,
				params map[string]any,
			) (proxias.ActionExecutionResult, error) {
				for _, name := range required {
					v := params[name] // nil when it is missing
					if v == nil || v == "" {
						c.record(t.id, params, true)
						return proxias.ActionExecutionResult{
							ActionID: t.id,
							Status:   "failed",
							Data: map[string]any{
								"error": "missing parameter: " + name,
							},
						}, nil
					}
				}
				c.record(t.id, params, false)
				data := map[string]any{"ok": true}
				if t.id == "crm_find_customer" {
					data = map[string]any{"found": false}
					if strings.Contains(strings.ToLower(fmt.Sprint(params["name"])), "globex") {
						data = map[string]any{
							"customer_id":   "C-204",
							"name":          "Globex Corporation",
							"contact_email": "ops@globex.example",
						}
					}
				}
				return proxias.ActionExecutionResult{
					ActionID: t.id,
					Status:   "completed",
					Data:     data,
				}, nil
			},
		}
	}
	return executor
}

// schemaOf is the JSON Schema of the parameters, and the names of the required
// ones.
func schemaOf(params []string) (map[string]any, []string) {
	properties := map[string]any{}
	required := []string{}
	for _, p := range params {
		parts := strings.SplitN(p, ":", 3)
		name := strings.TrimPrefix(parts[0], "*")
		property := map[string]any{"type": parts[1], "description": parts[2]}
		if parts[1] == "strings" {
			property = map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": parts[2],
			}
		}
		properties[name] = property
		if strings.HasPrefix(parts[0], "*") {
			required = append(required, name)
		}
	}
	return map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}, required
}
