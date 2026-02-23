'''
LLM class to handle interactions with the language model:
    • generating responses
    • loading and saving conversation history
'''
import requests
import json
import re

class LLM:

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        api_url: str = "http://localhost:11434/api/chat"
    ):
        self.model_name = model_name
        self.api_url = api_url

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
    "eval_count": <int>
    }
    '''

    def ask_llm(self, prompt):
        """Query LLM with structured JSON output constraint - BULLETPROOF VERSION."""
        
        structured_prompt = f"""{prompt}

    CRITICAL: Return ONLY valid JSON with NO extra text.
    Format:
    {{"traits": ["trait 1", "trait 2", "trait 3", "trait 4", "trait 5"]}}"""

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": structured_prompt}],
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            
            # ============ STEP 1: EXTRACT JSON PORTION ============
            
            # Find first { and strip preamble
            start_idx = content.find('{')
            if start_idx == -1:
                raise ValueError("No JSON object found")
            
            json_part = content[start_idx:]
            
            # Remove markdown code blocks
            json_part = json_part.replace('```json', '').replace('```', '').strip()
            
            # ============ STEP 2: BALANCE BRACKETS ============
            
            open_count = json_part.count('{')
            close_count = json_part.count('}')
            
            if open_count > close_count:
                json_part = json_part + ('}' * (open_count - close_count))
            
            # ============ STEP 3: AGGRESSIVE CLEANUP ============
            
            # Remove trailing commas and spaces before closing brackets
            json_part = re.sub(r',\s*([}\]])', r'\1', json_part)
            
            # Remove any trailing spaces/commas after the last quote in arrays
            # This fixes: "trait" ]  or  "trait",  ]
            json_part = re.sub(r'"\s+\]', '"]', json_part)
            json_part = re.sub(r'",\s*\]', '"]', json_part)
            
            # Normalize whitespace inside strings (in case there are weird unicode spaces)
            # First, extract and clean each trait individually
            def clean_trait_in_json(match):
                trait = match.group(1)
                # Remove any control characters or weird unicode
                cleaned = ''.join(char for char in trait if char.isprintable() or char.isspace())
                cleaned = cleaned.strip()
                return f'"{cleaned}"'
            
            json_part = re.sub(r'"([^"]*)"', clean_trait_in_json, json_part)
            
            # ============ STEP 4: PARSE ============
            
            try:
                parsed = json.loads(json_part)
                traits = parsed.get("traits", [])
                
            except json.JSONDecodeError as first_error:
                print(f"⚠️  First parse failed: {first_error}")
                print(f"📝 Attempting array extraction...")
                
                # Fallback: Extract just the array content
                array_pattern = r'"traits"\s*:\s*\[(.*?)\]'
                array_match = re.search(array_pattern, json_part, re.DOTALL)
                
                if array_match:
                    array_content = array_match.group(1).strip()
                    
                    # Clean the array content
                    array_content = re.sub(r',\s*$', '', array_content)  # Remove trailing comma
                    
                    reconstructed = f'{{"traits": [{array_content}]}}'
                    
                    try:
                        parsed = json.loads(reconstructed)
                        traits = parsed.get("traits", [])
                        print(f"✓ Successfully extracted via array pattern")
                    except json.JSONDecodeError as second_error:
                        print(f"⚠️  Array extraction also failed: {second_error}")
                        raise first_error
                else:
                    raise first_error
            
            # ============ STEP 5: VALIDATE & CLEAN ============
            
            if not isinstance(traits, list):
                raise ValueError("Traits must be a list")
            
            if len(traits) == 0:
                raise ValueError("Traits list is empty")
            
            # Clean each trait
            cleaned_traits = []
            for t in traits:
                if isinstance(t, str):
                    # Remove any non-printable characters
                    cleaned = ''.join(char for char in t if char.isprintable() or char.isspace())
                    cleaned = cleaned.strip()
                    
                    # Skip JSON keys
                    if cleaned and cleaned.lower() not in ['traits', 'trait']:
                        cleaned_traits.append(cleaned)
            
            if len(cleaned_traits) == 0:
                raise ValueError("No valid traits after cleaning")
            
            return cleaned_traits[:5]
            
        except (json.JSONDecodeError, ValueError) as e:
            # ============ FINAL FALLBACK: REGEX ============
            print(f"⚠️  All parsing failed: {e}")
            print(f"📝 Content: {content[:300]}...")
            
            # Extract quoted strings
            trait_pattern = r'"([^"]{3,150})"'
            found_traits = re.findall(trait_pattern, content)
            
            # Clean and filter
            excluded = {'traits', 'trait', 'skills', 'skill'}
            cleaned = []
            for t in found_traits:
                clean_t = ''.join(char for char in t if char.isprintable() or char.isspace()).strip()
                if clean_t and clean_t.lower() not in excluded:
                    cleaned.append(clean_t)
            
            if len(cleaned) >= 3:
                print(f"✓ Recovered {len(cleaned)} traits via regex")
                return cleaned[:5]
            
            return {"error": "json_parse_failed", "raw_content": content[:500]}
        
        except requests.exceptions.RequestException as e:
            return {"error": "api_request_failed", "message": str(e)}
        
        
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