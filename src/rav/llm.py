'''
LLM class to handle interactions with the language model:
    • generating responses
    • extracting traits from responses
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

    def ask_llm(self, prompt,
        n: int = None,
        experiment_id: str = None,
        job_code: str = None,
        job_title: str = None,
        template_type: str = None,
        gender_condition: str = None
        )-> dict:
        """Query LLM and return extracted traits with response metadata.

            Args:
                prompt:           Prompt text to send to the LLM
                n:                Number of traits expected in response
                experiment_id:    Experiment identifier for response ID
                job_code:         O*NET-SOC code for response ID
                template_type:    Prompt template type (T1/T2) for response ID
                gender_condition: Gender condition for response ID

            Returns:
                Dict with response_id, traits, and metadata
            """
        
        # GENERATE RESPONSE ID
        response_id = (
            f"{experiment_id}_{job_code}_{template_type}_{gender_condition}"
            if all([experiment_id, job_code, template_type, gender_condition])
            else None
        )

        structured_prompt = f"""{prompt}

        CRITICAL: Return ONLY valid JSON with NO extra text.
        Format:
        {{"traits": ["trait 1", "trait 2", ..., "trait {n}"]}}
        """

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
                return f'"{cleaned.strip()}"'
            
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
            excluded = {'traits', 'trait', 'skills', 'skill'}
            cleaned_traits = []
            for t in traits:
                if isinstance(t, str):
                    # Remove any non-printable characters
                    cleaned = ''.join(char for char in t if char.isprintable() or char.isspace()).strip()
                    
                    # Skip JSON keys
                    if cleaned and cleaned.lower() not in excluded:
                        cleaned_traits.append(cleaned)
            
            if len(cleaned_traits) == 0:
                raise ValueError("No valid traits after cleaning")
            
            return {
                'response_id': response_id,
                'job_code': job_code,
                'job_title': job_title,
                'template_type': template_type,
                'gender_condition': gender_condition,
                'experiment_id': experiment_id,
                'traits': cleaned_traits[:n],
                'n_traits': len(cleaned_traits[:n]),
                'status': 'success'
            }
            
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
                return {
                    'response_id': response_id,
                    'job_code': job_code,
                    'job_title': job_title,
                    'template_type': template_type,
                    'gender_condition': gender_condition,
                    'experiment_id': experiment_id,
                    'traits': cleaned[:n],
                    'n_traits': len(cleaned[:n]),
                    'status': 'recovered'
                }
            
            return {
                'response_id': response_id,
                'job_code': job_code,
                'job_title': job_title,
                'template_type': template_type,
                'gender_condition': gender_condition,
                'experiment_id': experiment_id,
                'traits': [],
                'n_traits': 0,
                'status': 'failed',
                'error': 'json_parse_failed',
                'raw_content': content[:500]
            }
        
        except requests.exceptions.RequestException as e:
            return {
                'response_id': response_id,
                'job_code': job_code,
                'job_title': job_title,
                'template_type': template_type,
                'gender_condition': gender_condition,
                'experiment_id': experiment_id,
                'traits': [],
                'n_traits': 0,
                'status': 'failed',
                'error': 'api_request_failed',
                'message': str(e)
            }