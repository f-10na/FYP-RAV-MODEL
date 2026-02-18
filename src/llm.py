'''
LLM class to handle interactions with the language model:
    • generating responses
    • loading and saving conversation history
'''
import requests
import sys
import os
import re


class LLM:

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        API_URL: str = "http://localhost:11434/api/chat"
    ):
        'initializes the LLM class with the specified model name and API URL'
        self.model_name = model_name
        self.API_URL = API_URL

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

    def ask_llm(prompt,model_name,API_URL):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False  # This prevents getting back multiple JSON chunks
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # Check for errors
            
            # Parse the JSON response
            data = response.json()
            content = data.get("message", {}).get("content", "")
            content = re.sub(r'[\*\#]', '', content)
                
            return content.strip()
        
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"


    """
    Extracts trait phrases from LLM output, removing obvious noise.
    """  
    def parse_llm_traits(raw_text):
    
        # Step 1: Split by lines
        lines = raw_text.split('\n')
        
        traits = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Step 2: Remove leading numbers, bullets, or hyphens
            line = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', line)
            
            # Step 3: Keep only text before colon (if colon exists)
            if ':' in line:
                line = line.split(':', 1)[0]
            
            # Step 4: Clean extra whitespace
            line = line.strip()
            
            if line:
                traits.append(line)
        
        return traits