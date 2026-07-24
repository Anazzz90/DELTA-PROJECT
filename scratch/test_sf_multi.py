import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")
url = "https://api.siliconflow.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

models = ["zai-org/GLM-4.5-Air", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1"]

for model in models:
    print(f"Calling {model}...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, please say 'OK' if you are working."}],
        "stream": False,
        "max_tokens": 50
    }
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed = time.time() - start
        print(f"  Status Code: {response.status_code}")
        print(f"  Time taken: {elapsed:.1f}s")
        if response.status_code != 200:
             print(f"  Response: {response.text}")
    except Exception as e:
        print(f"  Failed: {e}")
