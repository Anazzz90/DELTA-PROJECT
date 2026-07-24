import os
import litellm
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")
print(f"API Key found: {api_key[:5]}...{api_key[-5:]}" if api_key else "No API key found")

model = "openai/deepseek-ai/DeepSeek-V3"
api_base = "https://api.siliconflow.cn/v1"

try:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        api_key=api_key,
        api_base=api_base,
        timeout=10
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
