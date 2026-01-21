'''
Use Ollama to load free local cloud LLM models.
Struture of the streamed JSON:
{
  "model": "<model_name>",
  "created_at": "<timestamp>",
  "message": {
    "role": "assistant",
    "content": "<partial_text>"
  },
  "done": <true/false>,
  "done_reason": "<stop/reason>",
  "total_duration": <nanoseconds>,
  "load_duration": <nanoseconds>,
  "prompt_eval_count": <int>,
  "eval_count": <int>,
  ...
}

'''

import requests

# Store the base API URL in a variable
API_URL = "http://localhost:11434/api/chat"

def ask_llm(model_name, prompt):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, json=payload)
    return response.text

# Usage
answer = ask_llm(prompt)
print(answer)