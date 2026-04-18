import os
import requests
from langfuse import observe
# Note: Langfuse CallbackHandler is exclusively for LangChain. Since we are using raw requests, 
# the @observe decorator is used instead to guarantee 1:1 telemetry tracking without the LangChain dependency.

_LLM_DISABLED = False

@observe(as_type="generation", name="llm_check")
def llm_check(prompt: str, session_id: str, callback_handler=None) -> str:
    global _LLM_DISABLED

    if os.getenv("FRAUD_PIPELINE_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return "NO"

    if _LLM_DISABLED:
        return "NO"

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        _LLM_DISABLED = True
        return "NO"

    timeout = float(os.getenv("OPENROUTER_TIMEOUT", "5"))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        _LLM_DISABLED = True
        print(f"LLM Connection Error: {e}")
        return "NO"
