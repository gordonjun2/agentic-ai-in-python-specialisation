'''
Now that you understand the agent loop and how to craft effective prompts, we can build a simple AI agent. 
This agent will be able to list files in a directory, read their content, and answer questions about them. 
We'll break down the agent loop—how it receives input, decides on actions, executes them, and updates its 
memory—step by step.

The agent loop is the backbone of our AI agent, enabling it to perform tasks by combining response generation, 
action execution, and memory updates in an iterative process. This section focuses on how the agent loop works 
and its role in making the agent dynamic and adaptive.

Here's the rewrite of the steps using the more descriptive and cohesive style from the original explanation:
1. Construct Prompt: Combine the agent's memory, user input, and system rules into a single prompt. This ensures 
    the LLM has all the context it needs to decide on the next action, maintaining continuity across iterations.
2. Generate Response: Send the constructed prompt to the LLM and retrieve a response. This response will guide 
    the agent's next step by providing instructions in a structured format.
3. Parse Response: Extract the intended action and its parameters from the LLM's output. The response must adhere 
    to a predefined structure (e.g., JSON format) to ensure it can be interpreted correctly.
4. Execute Action: Use the extracted action and its parameters to perform the requested task with the appropriate 
    tool. This could involve listing files, reading content, or printing a message.
5. Convert Result to String: Format the result of the executed action into a string. This allows the agent to store 
    the result in its memory and provide clear feedback to the user or itself.
6. Continue Loop?: Evaluate whether the loop should continue based on the current action and results. The loop may 
    terminate if a “terminate” action is specified or if the agent has completed the task.

The agent iterates through this loop, refining its behavior and adapting its actions until it reaches a stopping 
condition. This process is what enables the agent to interact dynamically and respond intelligently to tasks.
'''

import sys
import os

current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(main_dir)

from config import *
from litellm import completion
from typing import List, Dict
import json

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(model="openai/gpt-4o",
                          messages=messages,
                          max_tokens=1024)
    return response.choices[0].message.content


def parse_action(response: str) -> Dict:
    """Parse the LLM response into a structured action dictionary."""
    try:
        response = extract_markdown_block(response, "action")
        response_json = json.loads(response)
        if "tool_name" in response_json and "args" in response_json:
            return response_json
        else:
            return {
                "tool_name": "error",
                "args": {
                    "message": "You must respond with a JSON tool invocation."
                }
            }
    except json.JSONDecodeError:
        return {
            "tool_name": "error",
            "args": {
                "message":
                "Invalid JSON response. You must respond with a JSON tool invocation."
            }
        }


agent_rules = [{
    "role":
    "system",
    "content":
    """
    You are an AI agent that can perform tasks by using available tools.

    Available tools:
    - list_files() -> List[str]: List all files in the current directory.
    - read_file(file_name: str) -> str: Read the content of a file.
    - terminate(message: str): End the agent loop and print a summary to the user.

    If a user asks about files, list them before reading.

    Every response MUST have an action.
    Respond in this format:

    ```action
    {
        "tool_name": "insert tool_name",
        "args": {...fill in any required arguments here...}
    }
    ```
    """
}]

memory = [{
    "role": "user",
    "content": "What files are in this directory?"
}, {
    "role":
    "assistant",
    "content":
    "```action\n{\"tool_name\":\"list_files\",\"args\":{}}\n```"
}, {
    "role": "user",
    "content": "[\"file1.txt\", \"file2.txt\"]"
}]

max_iterations = 3

# The Agent Loop
while iterations < max_iterations:

    # 1. Construct prompt: Combine agent rules with memory
    prompt = agent_rules + memory

    # 2. Generate response from LLM
    print("Agent thinking...")
    response = generate_response(prompt)
    print(f"Agent response: {response}")

    # 3. Parse response to determine action
    action = parse_action(response)

    result = "Action executed"

    if action["tool_name"] == "list_files":
        result = {"result": list_files()}
    elif action["tool_name"] == "read_file":
        result = {"result": read_file(action["args"]["file_name"])}
    elif action["tool_name"] == "error":
        result = {"error": action["args"]["message"]}
    elif action["tool_name"] == "terminate":
        print(action["args"]["message"])
        break
    else:
        result = {"error": "Unknown action: " + action["tool_name"]}

        print(f"Action result: {result}")

    # 5. Update memory with response and results
    memory.extend([{
        "role": "assistant",
        "content": response
    }, {
        "role": "user",
        "content": json.dumps(result)
    }])

    # 6. Check termination condition
    if action["tool_name"] == "terminate":
        break

    iterations += 1
