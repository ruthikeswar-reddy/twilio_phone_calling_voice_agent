# CSA Agent API Test Scenarios

## Quick curl Tests

### 1. Basic Flow (3 turns → Ticket)
```bash
# Turn 1
curl -X POST "http://localhost:8001/chat/csa" -H 'Content-Type: application/json' -d '{"message": "VDI not working"}'

# Turn 2 (same session)
curl -X POST "http://localhost:8001/chat/csa" -H 'Content-Type: application/json' -d '{"message": "Stuck on loading screen after login", \"session_id\": \"test1\"}' -s | grep -E '(data:|ticket)'

# Turn 3: Confirm → TICKET CREATED
curl -X POST "http://localhost:8001/chat/csa" -H 'Content-Type: application/json' -d '{"message": "Yes correct", \"session_id\": \"test1\"}' -s | grep ticket
```

### 2. Employee Data
```bash
curl -X POST "http://localhost:8001/chat/csa" -d '{"message": "EM123 john@company.com Hyderabad printer not working"}'
# → Shows summary with extracted fields
```

### 3. Edit Flow
```bash
curl -X POST "http://localhost:8001/chat/csa" -d '{"message": "Network slow", \"session_id\": \"edit1\"}'
curl -X POST "http://localhost:8001/chat/csa" -d '{"message": "Change location to Bangalore", \"session_id\": \"edit1\"}'
# → Updates location field
```

## Automated Tests
```bash
python test_agent_api.py
```

## Browser Test (HTML)
Save as `test_chat.html`:
```html
<!DOCTYPE html>
<html>
<body>
  <input id='input' placeholder='Type message' style='width:400px'>
  <button onclick='send()'>Send</button>
  <div id='output'></div>
  <script>
  let sessionId = 'browser-test';
  async function send() {
    const msg = document.getElementById('input').value;
    const resp = await fetch('/chat/csa', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, session_id: sessionId})
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      chunk.split('\\n\\n').forEach(line => {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.text) document.getElementById('output').innerHTML += data.text;
          } catch(e) {}
        }
      });
    }
    document.getElementById('input').value = '';
  }
  </script>
</body>
</html>
```
Open → Chat interface ready!

## Expected Agent Responses

| Input | Expected Response |
|-------|------------------|
| \"VDI not working\" | \"Could you explain the issue in detail?\" |
| Employee ID only | Extracts ID, asks for description |
| Full ticket data | Shows summary → \"Is everything correct?\" |
| \"Yes\" | Creates ticket → \"Ticket ID: INCxxxxx\" |
| \"Change email\" | Asks for new email |
```

