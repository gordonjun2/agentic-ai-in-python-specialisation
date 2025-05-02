import sys
import os

current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(main_dir)

from config import *
import json
import time
import traceback
import inspect
from litellm import completion
from dataclasses import dataclass, field
from typing import get_type_hints, List, Callable, Dict, Any
import uuid
import tiktoken
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import reduce

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

tools = {}
tools_by_tag = {}


def count_tokens(messages):
    encoding = tiktoken.get_encoding("cl100k_base")  # The GPT-4 tokenizer
    total_tokens = 0

    for message in messages:
        total_tokens += len(encoding.encode(message["content"]))

    return total_tokens


def recursive_json_loads(value):
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return recursive_json_loads(loaded)
        except json.JSONDecodeError:
            return value
    elif isinstance(value, dict):
        return {k: recursive_json_loads(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [recursive_json_loads(item) for item in value]
    else:
        return value


def has_named_parameter(func, name: str) -> bool:
    """
    Check if the given function has a parameter with the specified name.
    
    Parameters:
        func: The function to inspect.
        name: The name of the parameter to check for.
        
    Returns:
        True if the parameter exists in the function signature, False otherwise.
    """
    try:
        sig = inspect.signature(func)
        return name in sig.parameters
    except:
        return False


def get_tool_metadata(func,
                      tool_name=None,
                      description=None,
                      parameters_override=None,
                      terminal=False,
                      tags=None):
    """Extract metadata while ignoring special parameters."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    args_schema = {"type": "object", "properties": {}, "required": []}

    for param_name, param in signature.parameters.items():
        # Skip special parameters - agent doesn't need to know about these
        if param_name in ["action_context", "action_agent"] or \
           param_name.startswith("_"):
            continue

        # Add regular parameters to the schema
        param_type = type_hints.get(param_name, str)
        args_schema["properties"][param_name] = {
            "type": "string"  # Simplified for example
        }

        if param.default == param.empty:
            args_schema["required"].append(param_name)

    return {
        "name": tool_name or func.__name__,
        "description": description or func.__doc__,
        "parameters": args_schema,
        "tags": tags or [],
        "terminal": terminal,
        "function": func
    }


def register_tool(tool_name=None,
                  description=None,
                  parameters_override=None,
                  terminal=False,
                  tags=None):
    """
    A decorator to dynamically register a function in the tools dictionary with its parameters, schema, and docstring.

    Parameters:
        tool_name (str, optional): The name of the tool to register. Defaults to the function name.
        description (str, optional): Override for the tool's description. Defaults to the function's docstring.
        parameters_override (dict, optional): Override for the argument schema. Defaults to dynamically inferred schema.
        terminal (bool, optional): Whether the tool is terminal. Defaults to False.
        tags (List[str], optional): List of tags to associate with the tool.

    Returns:
        function: The wrapped function.
    """

    def decorator(func):
        # Use the reusable function to extract metadata
        metadata = get_tool_metadata(func=func,
                                     tool_name=tool_name,
                                     description=description,
                                     parameters_override=parameters_override,
                                     terminal=terminal,
                                     tags=tags)

        # Register the tool in the global dictionary
        tools[metadata["name"]] = {
            "description": metadata["description"],
            "parameters": metadata["parameters"],
            "function": metadata["function"],
            "terminal": metadata["terminal"],
            "tags": metadata["tags"] or []
        }

        for tag in metadata["tags"]:
            if tag not in tools_by_tag:
                tools_by_tag[tag] = []
            tools_by_tag[tag].append(metadata["name"])

        return func

    return decorator


@dataclass
class Prompt:
    messages: List[Dict] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    metadata: dict = field(
        default_factory=dict)  # Fixing mutable default issue


def generate_response(prompt: Prompt) -> str:
    """Call LLM to get response"""

    messages = prompt.messages
    tools = prompt.tools

    result = None

    token_count = count_tokens(messages)
    print(f"Total tokens: {token_count}")

    if not tools:
        response = completion(model="openai/gpt-4o",
                              messages=messages,
                              max_tokens=1024)
        result = response.choices[0].message.content
    else:
        response = completion(model="openai/gpt-4o",
                              messages=messages,
                              tools=tools,
                              max_tokens=1024)

        if response.choices[0].message.tool_calls:
            tool = response.choices[0].message.tool_calls[0]
            result = {
                "tool": tool.function.name,
                "args": json.loads(tool.function.arguments),
            }
            result = json.dumps(result)
        else:
            result = response.choices[0].message.content

    return result


@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str


class Action:

    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)


class ActionRegistry:

    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> [Action, None]:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())


class ActionContext:

    def __init__(self, properties: Dict = None):
        self.context_id = str(uuid.uuid4())
        self.properties = properties or {}

    def get(self, key: str, default=None):
        return self.properties.get(key, default)

    def get_memory(self):
        return self.properties.get("memory", None)


class Memory:

    def __init__(self):
        self.items = []  # Basic conversation histor

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory


class Environment:

    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }


class PythonEnvironment(Environment):

    def execute_action(self, action_context: ActionContext, action: Action,
                       args: dict) -> dict:
        """Execute an action with automatic dependency injection."""
        try:
            # Create a copy of args to avoid modifying the original
            args_copy = args.copy()

            # If the function wants action_context, provide it
            if has_named_parameter(action.function, "action_context"):
                args_copy["action_context"] = action_context

            # Inject properties from action_context that match _prefixed parameters
            for key, value in action_context.properties.items():
                param_name = "_" + key
                if has_named_parameter(action.function, param_name):
                    args_copy[param_name] = value

            # Execute the function with injected dependencies
            result = action.execute(**args_copy)
            return self.format_result(result)
        except Exception as e:
            return {"tool_executed": False, "error": str(e)}


class AgentLanguage:

    def __init__(self):
        pass

    def construct_prompt(self, actions: List[Action], environment: Environment,
                         goals: List[Goal], memory: Memory) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


class AgentFunctionCallingActionLanguage(AgentLanguage):

    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]) -> List:
        # Map all goals to a single string that concatenates their description
        # and combine into a single message of type system
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join(
            [f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals])
        return [{"role": "system", "content": goal_instructions}]

    def format_memory(self, memory: Memory) -> List:
        """Generate response from language model"""
        # Map all environment results to a role:user messages
        # Map all assistant messages to a role:assistant messages
        # Map all user messages to a role:user messages
        items = memory.get_memories()
        mapped_items = []
        for item in items:

            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> [List, List]:
        """Generate response from language model"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    # Include up to 1024 characters of the description
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            } for action in actions
        ]

        return tools

    def construct_prompt(self, actions: List[Action], environment: Environment,
                         goals: List[Goal], memory: Memory) -> Prompt:

        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(self, prompt: Prompt, response: str,
                                         traceback: str, error: Any,
                                         retries_left: int) -> Prompt:

        return prompt

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format by extracting the ```json block"""

        try:
            return recursive_json_loads(response)

        except Exception as e:
            return {"tool": "terminate", "args": {"message": response}}


class PythonActionRegistry(ActionRegistry):

    def __init__(self, tags: List[str] = None, tool_names: List[str] = None):
        super().__init__()

        self.terminate_tool = None

        for tool_name, tool_desc in tools.items():
            if tool_name == "terminate":
                self.terminate_tool = tool_desc

            if tool_names and tool_name not in tool_names:
                continue

            tool_tags = tool_desc.get("tags", [])
            if tags and not any(tag in tool_tags for tag in tags):
                continue

            self.register(
                Action(name=tool_name,
                       function=tool_desc["function"],
                       description=tool_desc["description"],
                       parameters=tool_desc.get("parameters", {}),
                       terminal=tool_desc.get("terminal", False)))

    def register_terminate_tool(self):
        if self.terminate_tool:
            self.register(
                Action(name="terminate",
                       function=self.terminate_tool["function"],
                       description=self.terminate_tool["description"],
                       parameters=self.terminate_tool.get("parameters", {}),
                       terminal=self.terminate_tool.get("terminal", False)))
        else:
            raise Exception("Terminate tool not found in tool registry")


class Capability:

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def init(self, agent, action_context: ActionContext) -> dict:
        """Called once when the agent starts running."""
        pass

    def start_agent_loop(self, agent, action_context: ActionContext) -> bool:
        """Called at the start of each iteration through the agent loop."""
        return True

    def process_prompt(self, agent, action_context: ActionContext,
                       prompt: Prompt) -> Prompt:
        """Called right before the prompt is sent to the LLM."""
        return prompt

    def process_response(self, agent, action_context: ActionContext,
                         response: str) -> str:
        """Called after getting a response from the LLM."""
        return response

    def process_action(self, agent, action_context: ActionContext,
                       action: dict) -> dict:
        """Called after parsing the response into an action."""
        return action

    def process_result(self, agent, action_context: ActionContext,
                       response: str, action_def: Action, action: dict,
                       result: any) -> any:
        """Called after executing the action."""
        return result

    def process_new_memories(self, agent, action_context: ActionContext,
                             memory: Memory, response, result,
                             memories: List[dict]) -> List[dict]:
        """Called when new memories are being added."""
        return memories

    def end_agent_loop(self, agent, action_context: ActionContext):
        """Called at the end of each iteration through the agent loop."""
        pass

    def should_terminate(self, agent, action_context: ActionContext,
                         response: str) -> bool:
        """Called to check if the agent should stop running."""
        return False

    def terminate(self, agent, action_context: ActionContext) -> dict:
        """Called when the agent is shutting down."""
        pass


class TimeAwareCapability(Capability):

    def __init__(self):
        super().__init__(name="Time Awareness",
                         description="Allows the agent to be aware of time")

    def init(self, agent, action_context: ActionContext) -> dict:
        """Set up time awareness at the start of agent execution."""
        # Get timezone from context or use default
        time_zone_name = action_context.get("time_zone", "Asia/Singapore")
        timezone = ZoneInfo(time_zone_name)

        # Get current time in specified timezone
        current_time = datetime.now(timezone)

        # Format time in both machine and human-readable formats
        iso_time = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
        human_time = current_time.strftime("%H:%M %A, %B %d, %Y")

        # Store time information in memory
        memory = action_context.get_memory()
        memory.add_memory({
            "type":
            "system",
            "content":
            f"""Right now, it is {human_time} (ISO: {iso_time}).
            You are in the {time_zone_name} timezone.
            Please consider the day/time, if relevant, when responding."""
        })

    # def process_prompt(self, agent, action_context: ActionContext,
    #                    prompt: Prompt) -> Prompt:
    #     """Update time information in each prompt."""
    #     time_zone_name = action_context.get("time_zone", "Asia/Singapore")
    #     current_time = datetime.now(ZoneInfo(time_zone_name))

    #     # Add current time to system message
    #     system_msg = (f"Current time: "
    #                   f"{current_time.strftime('%H:%M %A, %B %d, %Y')} "
    #                   f"({time_zone_name})\n\n")

    #     # Add to existing system message or create new one
    #     messages = prompt.messages
    #     if messages and messages[0]["role"] == "system":
    #         messages[0]["content"] = system_msg + messages[0]["content"]
    #     else:
    #         messages.insert(0, {"role": "system", "content": system_msg})

    #     return Prompt(messages=messages)


class EnhancedTimeAwareCapability(TimeAwareCapability):

    def process_action(self, agent, action_context: ActionContext,
                       action: dict) -> dict:
        """Add timing information to action results."""
        # Add execution time to action metadata
        action["execution_time"] = datetime.now(
            ZoneInfo(action_context.get("time_zone",
                                        "Asia/Singapore"))).isoformat()
        return action

    def process_result(self, agent, action_context: ActionContext,
                       response: str, action_def: Action, action: dict,
                       result: any) -> any:
        """Add duration information to results."""
        if isinstance(result, dict):
            result["action_duration"] = (
                datetime.now(ZoneInfo(action_context.get("time_zone"))) -
                datetime.fromisoformat(
                    action["execution_time"])).total_seconds()
        return result


class Agent:

    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment,
                 capabilities: List[Capability] = [],
                 max_iterations: int = 10,
                 max_duration_seconds: int = 180):
        """
        Initialize an agent with its core GAME components and capabilities.

        Goals, Actions, Memory, and Environment (GAME) form the core of the agent,
        while capabilities provide ways to extend and modify the agent's behavior.

        Args:
            goals: What the agent aims to achieve
            agent_language: How the agent formats and parses LLM interactions
            action_registry: Available tools the agent can use
            generate_response: Function to call the LLM
            environment: Manages tool execution and results
            capabilities: List of capabilities that extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds

    def construct_prompt(self, action_context: ActionContext,
                         goals: List[Goal], memory: Memory,
                         actions: ActionRegistry) -> Prompt:
        """Build prompt with memory context"""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory)

    def get_action(self, response):
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """
        Update memory with the agent's decision and the environment's response.
        """
        new_memories = [{
            "type": "assistant",
            "content": response
        }, {
            "type": "environment",
            "content": json.dumps(result)
        }]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, action_context: ActionContext,
                              full_prompt: Prompt) -> str:
        response = self.generate_response(full_prompt)
        return response

    def handle_agent_response(self, action_context: ActionContext,
                              response: str) -> dict:
        """Handle action without dependency management."""
        action_def, base_action = self.get_action(response)

        # print('Action Definition:')
        # print(action_def)
        # print('Base Action:')
        # print(base_action)

        # Process action with capabilities
        action = reduce(lambda a, c: c.process_action(self, action_context, a),
                        self.capabilities, base_action)

        # print('Action:')
        # print(action)

        base_result = self.environment.execute_action(action_context,
                                                      action_def,
                                                      action["args"])

        # print('Base Result:')
        # print(base_result)

        # Process result with capabilities
        result = reduce(
            lambda r, c: c.process_result(self, action_context, response,
                                          action_def, action, r),
            self.capabilities, base_result)

        # print('Result:')
        # print(result)

        return result

    def run(self,
            user_input: str,
            memory=None,
            max_iterations: int = 50,
            action_context_props=None) -> Memory:
        """
        Execute the GAME loop for this agent with a maximum iteration limit.
        """
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        # Create context with all necessary resources
        action_context = ActionContext({
            'memory': memory,
            'llm': self.generate_response,
            **action_context_props
        })

        # Initialize capabilities
        for capability in self.capabilities:
            capability.init(self, action_context)

        for _ in range(max_iterations):

            # Start of loop capabilities
            can_start_loop = reduce(
                lambda a, c: c.start_agent_loop(self, action_context),
                self.capabilities, False)

            if can_start_loop:
                # Construct a prompt that includes the Goals, Actions, and the current Memory
                base_prompt = self.construct_prompt(action_context, self.goals,
                                                    memory, self.actions)

                # print('Base Prompt:')
                # print(base_prompt)

                # Construct prompt with capability modifications
                prompt = reduce(
                    lambda p, c: c.process_prompt(self, action_context, p),
                    self.capabilities, base_prompt)

                # print('Prompt:')
                # print(prompt)

                print("Agent thinking...")
                # Generate a response from the agent
                base_response = self.prompt_llm_for_action(
                    action_context, prompt)

                # print('Base Response:')
                # print(base_response)

                # Process response with capabilities
                response = reduce(
                    lambda r, c: c.process_response(self, action_context, r),
                    self.capabilities, base_response)

                print(f"Agent Decision: {response}")

                # Determine which action the agent wants to execute and execute the action in the environment
                result = self.handle_agent_response(action_context, response)
                print(f"Action Result: {result}")

                # Update the agent's memory with information about what happened
                self.update_memory(memory, response, result)

                # End of loop capabilities
                for capability in self.capabilities:
                    capability.end_agent_loop(self, action_context)

                # Check if the agent has decided to terminate
                if self.should_terminate(response):
                    break

            else:
                print("Agent loop not started due to capability conditions.")
                break

        return memory


def prompt_llm_for_json(action_context: ActionContext, schema: dict,
                        prompt: str):
    """
    Have the LLM generate JSON in response to a prompt. Always use this tool when you need structured data out of the LLM.
    This function takes a JSON schema that specifies the structure of the expected JSON response.
    
    Args:
        schema: JSON schema defining the expected structure
        prompt: The prompt to send to the LLM
        
    Returns:
        A dictionary matching the provided schema with extracted information
    """
    generate_response = action_context.get("llm")

    # Try up to 3 times to get valid JSON
    for i in range(3):
        try:
            # Send prompt with schema instruction and get response
            response = generate_response(
                Prompt(messages=[{
                    "role":
                    "system",
                    "content":
                    f"You MUST produce output that adheres to the following JSON schema:\n\n{json.dumps(schema, indent=4)}. Output your JSON in a ```json markdown block."
                }, {
                    "role": "user",
                    "content": prompt
                }]))

            # Check if the response has json inside of a markdown code block
            if "```json" in response:
                # Search from the front and then the back
                start = response.find("```json")
                end = response.rfind("```")
                response = response[start + 7:end].strip()

            # Parse and validate the JSON response
            return json.loads(response)

        except Exception as e:
            if i == 2:  # On last try, raise the error
                raise e
            print(f"Error generating response: {e}")
            print("Retrying...")


@register_tool(tags=["scheduling", "meetings"])
def create_meeting_data(action_context: ActionContext,
                        document_text: str) -> dict:
    """
    Extract structured meeting scheduling instructions from user text.

    This tool parses natural language instructions to identify key information needed
    to plan and schedule a meeting, such as preferred time windows, duration,
    participants, and timezone.

    Args:
        document_text: The raw user instruction text for scheduling a meeting.

    Returns:
        A dictionary containing the extracted meeting planning information.
    """
    meeting_request_schema = {
        "type": "object",
        "required": ["duration_minutes", "date", "time_range", "attendees"],
        "properties": {
            "duration_minutes": {
                "type": "integer",
                "description": "Duration of the meeting in minutes"
            },
            "date": {
                "type": "string",
                "format": "date",
                "description": "The date the meeting should occur"
            },
            "time_range": {
                "type": "object",
                "required": ["start", "end"],
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Earliest preferred start time"
                    },
                    "end": {
                        "type": "string",
                        "description": "Latest preferred end time"
                    }
                }
            },
            "attendees": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of people to invite to the meeting"
            },
            "timezone": {
                "type": "string",
                "description":
                "Timezone of the meeting (e.g., 'Asia/Singapore')"
            },
            "title": {
                "type": "string",
                "description": "Optional title for the meeting"
            }
        }
    }

    extraction_prompt = f"""
        You are a meeting scheduling assistant. Extract the following structured information
        from the user instructions below:

        - Desired meeting date (use today's date if 'today' is mentioned)
        - Time range to schedule within (e.g., 14:00 to 17:00)
        - Meeting duration in minutes
        - List of attendees
        - Timezone if mentioned
        - Optional meeting title

        Think step-by-step. Return only valid JSON that matches the schema.

        <user_request>
        {document_text}
        </user_request>
    """

    return prompt_llm_for_json(action_context=action_context,
                               schema=meeting_request_schema,
                               prompt=extraction_prompt)


@register_tool(tags=["storage", "meetings"])
def store_meeting_data(action_context: ActionContext,
                       meeting_data: dict) -> dict:
    """
    Store a scheduled meeting in the meeting database. If a meeting with the same ID
    or time/title combo exists, it will be updated.

    Args:
        meeting_data: A dictionary containing structured meeting info. Example:
        {
            "title": "Team Sync",
            "date": "2024-05-02",
            "start_time": "14:00",
            "end_time": "14:30",
            "timezone": "Asia/Singapore",
            "attendees": ["Alice", "Bob", "Charlie"],
            "location": "Zoom",
            "description": "Discuss project status"
        }

    Returns:
        A dictionary confirming storage with metadata
    """

    # Get storage dictionary from action_context or initialize
    storage = action_context.get("meeting_storage", {})

    # Use date + time + title as a basic unique key
    unique_id = f"{meeting_data['date']}T{meeting_data['start_time']}_{meeting_data['title']}"
    storage[unique_id] = meeting_data

    return {
        "status": "success",
        "message":
        f"Stored meeting '{meeting_data['title']}' at {meeting_data['start_time']} on {meeting_data['date']}",
        "meeting_id": unique_id,
        "attendees": meeting_data["attendees"]
    }


@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    """Terminates the agent's execution with a final message.

    Args:
        message: The final message to return before terminating

    Returns:
        The message with a termination note appended
    """
    return f"{message}\nTerminating..."


def create_scheduler_agent():
    # Create action registry with invoice tools
    action_registry = PythonActionRegistry()

    # Define goals
    goals = [
        Goal(priority=1,
             name="schedule_meetings",
             description="""Schedule meetings efficiently by:
            1. Finding times that work for all attendees
            2. Handling any scheduling conflicts
            3. Once the meeting is scheduled, terminate the agent""")
    ]

    # Define agent environment
    environment = PythonEnvironment()

    return Agent(goals=goals,
                 agent_language=AgentFunctionCallingActionLanguage(),
                 action_registry=action_registry,
                 generate_response=generate_response,
                 environment=environment,
                 capabilities=[EnhancedTimeAwareCapability()])


def main():
    user_text = """
    Schedule a 30-minute team sync meeting today between 8 PM and 11 PM Singapore time.
    Include Alice, Bob, and Charlie. Prioritize a time that works for everyone and send out calendar invites.
    """

    # Create an agent instance
    agent = create_scheduler_agent()

    # Process the invoice
    final_memory = agent.run(user_text,
                             action_context_props={
                                 'time_zone': 'Asia/Singapore',
                                 "meeting_storage": {}
                             })

    print('\n')
    print(final_memory.get_memories())


if __name__ == "__main__":
    main()
