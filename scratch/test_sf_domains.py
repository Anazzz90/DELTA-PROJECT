import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")

# Try both domains
urls = [
    "https://api.siliconflow.cn/v1/chat/completions",
    "https://api.siliconflow.com/v1/chat/completions"
]

for url in urls:
    print(f"Testing {url}...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"  Status Code: {response.status_code}")
        print(f"  Response: {response.text}")
    except Exception as e:
        print(f"  Failed: {e}")
