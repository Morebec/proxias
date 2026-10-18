# Changelog

## 1.0.0 (2026-10-18)


### ⚠ BREAKING CHANGES

* every Workflow identifier is renamed to Action: WorkflowExecutionEngine (WorkflowDescriptors, ExecuteWorkflows) is ActionExecutionEngine (ActionDescriptors, ExecuteActions); WorkflowExecutionSynthesizer[Func], WorkflowExecutionSynthesis, WorkflowDescriptor, WorkflowExecutionResult, WorkflowExecutionIntent, WorkflowExecutionIntentSelector[Func] and WorkflowExecutionIntentParameterExtractor[Func] are their Action counterparts; Intent.WorkflowExecutions is Intent.ActionExecutions; WorkflowID is ActionID and its JSON field workflow_id is action_id; NewWorkflowInterceptionMiddleware, WithWorkflowDescriptors and WorkflowDescriptorsFrom are NewActionInterceptionMiddleware, WithActionDescriptors and ActionDescriptorsFrom. In the adapters, NewWorkflowExecutionIntentSelector, NewWorkflowExecutionIntentParameterExtractor and NewWorkflowExecutionSynthesizer are renamed likewise, and their prompts and the JSON the models read and answer say "actions" and "action_id".

### Features

* Add OpenAI adapter ([6e4797c](https://github.com/Morebec/proxias/commit/6e4797c64b07f3d794de9fc678a94063a91277b2))
* add the proxias base sources ([87a7117](https://github.com/Morebec/proxias/commit/87a7117f6eb54e4bd318851d8b5e2c293a19ba4c))
* add the proxias interceptor ([d2ed92c](https://github.com/Morebec/proxias/commit/d2ed92cf6d751d741e7d00f49fb0a3edf4aa394c))
* add the response builder, and read the sandbox key from the environment ([7de8474](https://github.com/Morebec/proxias/commit/7de8474e169074f4c40ca846bacacf68b0125fe1))
* implement the control plane and tidy general architecture ([30ac2fb](https://github.com/Morebec/proxias/commit/30ac2fb1de4601b3cf938214f810591534f007d1))
* pass the provider's error responses through the Anthropic reasoning engine ([1ab10f6](https://github.com/Morebec/proxias/commit/1ab10f64f9ce6d483cb129ba05e8ff545316ac86))
* rename workflows to actions ([#3](https://github.com/Morebec/proxias/issues/3)) ([9673cb2](https://github.com/Morebec/proxias/commit/9673cb2bc0f06c857938d4e2a67b2f0e34b06c81))
