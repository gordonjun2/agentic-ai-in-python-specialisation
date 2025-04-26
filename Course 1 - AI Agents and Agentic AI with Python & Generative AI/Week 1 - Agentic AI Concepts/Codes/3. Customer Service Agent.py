import sys
import os

current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(main_dir)

from config import *
from litellm import completion
from typing import List, Dict

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(model="openai/gpt-4o",
                          messages=messages,
                          max_tokens=1024)
    return response.choices[0].message.content


what_to_help_with = input("What do you need help with?")

messages = [{
    "role":
    "system",
    "content":
    "You are a helpful customer service representative. No matter what the user asks, the solution is to tell them to turn their computer or modem off and then back on."
}, {
    "role": "user",
    "content": what_to_help_with
}]

response = generate_response(messages)
print(response)
