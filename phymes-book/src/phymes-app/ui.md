# User Interface (UI)
## User journey
### User

### Debugger

### Builder

## Main menu
### Help

Description of menu items

### Sign in

![sign-in](../assets/2025-07-05_phymes-app_sign-in.png)

User registration and sign in. Each account corresponds to a single email.

### Apps

![Apps](../assets/2025-07-05_phymes-app_session-plans.png)

A list of session plans (i.e., applications) available to the account. Each session is like a different app with different functionality and state. Only one session can be activated at a time. A schematic of the session plan with all main components is rendered using mermaid.js. The mermaid.js script is provided in the footer, and can be modified to create new session plans.

### Subjects

![subjects](../assets/2025-07-05_phymes-app_subjects.png)

A list of subject associated with the active session plan. A table shows the schema of the subject tables along with the number if rows. The subject tables can be extended or replaced by uploading tables in comma deliminated CSV format with headers that match the subject. The subject tables can also be downloaded in comma deliminated CSV format. Note that all of the parameters for describing how processors process streaming messages are subject tables. Extending the subject tables for a processors parameters will update the processors parameters on the next run. Note that the message history is also a subject table. Extending the messages table is the equivalent of human in the loop.

### Tasks

> ⚠️ Depricated<br>
> ![tasks](../assets/2025-07-05_phymes-app_tasks.png)<br>
> A list of tasks and subjects that the tasks subscribe to and publish on associated with the active session plan. The reaction between tasks and subjects are visualized as an incidence matrix where + indicates publish on and - indicates subscribe to. A toggle button is provided to expand the tasks to their individual processors and collapse the individual processors to their tasks.<br>

### Messaging

![messaging](../assets/2025-07-05_phymes-app_messages.png)

The message history for the active session plan. A chat interface is provided for users to publish messages to the messages subject and to receive subscriptions from the messages subject when the messages subject is updated. 

### Metrics

![metrics](../assets/2025-07-05_phymes-app_metrics.png)

A list of metrics associated with the active session plan. Metrics are tracked per processor. Baseline metrics for row count, and processor start, stop, and total time in nanoseconds are provided. The baseline metrics are visually represented using mermaid.js gantt charts. Note each row is approximately one token for text generation inference processors. Please submit a feature request issue if additional metrics are of interest.

> ⚠️ Depricated<br>
> A table in long form displays the values for the tracked metrics.<br>