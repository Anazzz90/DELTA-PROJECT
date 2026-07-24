import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")
url = "https://api.siliconflow.com/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}"
}

try:
    response = requests.get(url, headers=headers)
    data = response.json()
    models = [m["id"] for m in data["data"]]
    models.sort()
    for m in models:
        print(m)
except Exception as e:
    print(f"Failed: {e}")
