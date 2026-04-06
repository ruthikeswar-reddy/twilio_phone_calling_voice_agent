# Voice Agent Workflow (Node-by-Node)

## Workflow 1: Normal Ticket Creation

### Turn 1
User: Hello, how can you help me?

Execution:
START → classify → greeting → END

Speak:
Hi! What issue are you facing today?

---

### Turn 2
User: I am not able to connect to my VDI

Execution:
START → classify → extract → ask → END

Internal (DO NOT SPEAK):
short_description = Unable to connect to VDI

Speak:
Could you explain what happens when you try?

---

### Turn 3
User: After login it keeps loading and does not open

Execution:
START → classify → extract → ask → END

Speak:
May I have your employee ID?

---

### Turn 4
User: EM1234

Execution:
extract → ask → END

Speak:
Could you share your company email?

---

### Turn 5
User: abc@company.com

Execution:
extract → ask → END

Speak:
Which location are you working from?

---

### Turn 6
User: Hyderabad

Execution:
extract → confirm → END

Speak:
(Read full summary of ticket)

---

### Turn 7
User: yes

Execution:
START → handle_confirm → create → END

Speak:
Your ticket has been created successfully. Ticket ID is INC0012345.

---

## Workflow 2: Update After Confirm

### Turn 7 (after summary)
User: I want to change my email

Execution:
START → handle_confirm → extract → ask → END

Speak:
Please provide your new email address

---

### Turn 8
User: new@mail.com

Execution:
START → extract → confirm → END

Speak:
(Read updated summary)

---

### Turn 9
User: yes

Execution:
handle_confirm → create → END

Speak:
Ticket created successfully. Ticket ID: INC0012345

---

## Voice Rules

### Only Speak:
result["messages"][-1].content

### Never Speak:
- JSON outputs
- slot dictionaries
- debug logs
- intermediate responses
