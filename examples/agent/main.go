package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/morebec/proxias/agentsdk"
)

func main() {
	agent := agentsdk.Agent{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	session, err := agent.Cogitate(ctx, agentsdk.Event{
		Type: agentsdk.EventTypeMessage,
		Source: agentsdk.EventSource{
			Type: agentsdk.EventSourceTypeUser,
			ID:   "john",
		},
		Payload: []byte(fmt.Sprintf(`Send an email to joe@boss.com saying "Hello Boss!".`)),
		Date:    time.Time{},
	})
	if err != nil {
		log.Fatalln("Error during cogitation:", err)
	}

	fmt.Println("History length:", len(session.Events))

	for _, event := range session.Events {
		fmt.Println(agentsdk.FormatEventToXMLish(event))
		//fmt.Println("Event Type:", event.Type)
		//fmt.Println("Event Source:", event.Source)
		//fmt.Println("Event Payload:", string(event.Payload))
		//fmt.Println("Event Date:", event.Date)
		//fmt.Println("Previous Event:", event.PreviousEvent)
		fmt.Println("-----")
	}
}
