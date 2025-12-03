from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import sys
import os
import time
import json

PORT = sys.argv[1]

API_KEY = "EMPTY"
BASE_URL = f"http://localhost:{PORT}/v1"
MODEL_ID = "Qwen3-8B"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def process(messages, rid=None, temperature=1.0, top_p=0.99):
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    content = ""
    print("---------------")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            chunk_content = chunk.choices[0].delta.content
            content += chunk_content
            print(chunk_content, end="", flush=True)
        if chunk.id is not None:
            rid = chunk.id
    print("\n---------------")
    return content, rid

def test_qwen():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    content, rid = process(messages)

if __name__ == "__main__":
    test_qwen()