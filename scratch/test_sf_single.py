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

# Testing DeepSeek-V3
payload = {
    "model": "deepseek-ai/DeepSeek-V3",
    "messages": [{"role": "user", "content": "Hello, please say 'OK' if you are working."}],
    "stream": False,
    "max_tokens": 50
}

print(f"Calling SiliconFlow DeepSeek-V3...")
start = time.time()
try:
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    elapsed = time.time() - start
    print(f"Status Code: {response.status_code}")
    print(f"Time taken: {elapsed:.1f}s")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
