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


class Agent:

    def __init__(self, goals: List[Goal], agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt],
                                             str], environment: Environment):
        """
        Initialize an agent with its core GAME components
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment

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
        action_def, action = self.get_action(response)

        result = self.environment.execute_action(action_context, action_def,
                                                 action["args"])
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

        for _ in range(max_iterations):
            # Construct a prompt that includes the Goals, Actions, and the current Memory
            prompt = self.construct_prompt(action_context, self.goals, memory,
                                           self.actions)

            print("Agent thinking...")
            # Generate a response from the agent
            response = self.prompt_llm_for_action(action_context, prompt)
            print(f"Agent Decision: {response}")

            # Determine which action the agent wants to execute and execute the action in the environment
            result = self.handle_agent_response(action_context, response)
            print(f"Action Result: {result}")

            # Update the agent's memory with information about what happened
            self.update_memory(memory, response, result)

            # Check if the agent has decided to terminate
            if self.should_terminate(response):
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


def prompt_expert(action_context: ActionContext, description_of_expert: str,
                  prompt: str) -> str:
    """
    Generate a response from an expert persona.
    
    The expert's background and specialization should be thoroughly described to ensure
    responses align with their expertise. The prompt should be focused on topics within
    their domain of knowledge.
    
    Args:
        description_of_expert: Detailed description of the expert's background and expertise
        prompt: The specific question or task for the expert
        
    Returns:
        The expert's response
    """
    generate_response = action_context.get("llm")
    response = generate_response(
        Prompt(messages=[{
            "role":
            "system",
            "content":
            f"Act as the following expert and respond accordingly: {description_of_expert}"
        }, {
            "role": "user",
            "content": prompt
        }]))
    return response


@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext,
                         document_text: str) -> dict:
    """
    Extract standardized invoice data from document text.

    This tool ensures consistent extraction of invoice information by using a fixed schema
    and specialized prompting for invoice understanding. It will identify key fields like
    invoice numbers, dates, amounts, and line items from any invoice format.

    Args:
        document_text: The text content of the invoice to process

    Returns:
        A dictionary containing the extracted invoice data in a standardized format
    """
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "total_amount"],
        "properties": {
            "invoice_number": {
                "type": "string"
            },
            "date": {
                "type": "string"
            },
            "total_amount": {
                "type": "number"
            },
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "address": {
                        "type": "string"
                    }
                }
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string"
                        },
                        "quantity": {
                            "type": "number"
                        },
                        "unit_price": {
                            "type": "number"
                        },
                        "total": {
                            "type": "number"
                        }
                    }
                }
            }
        }
    }

    # Create a focused prompt for invoice extraction
    extraction_prompt = f"""
            You are an expert invoice analyzer. Extract invoice information accurately and 
            thoroughly. Pay special attention to:
            - Invoice numbers (look for 'Invoice #', 'No.', 'Reference', etc.)
            - Dates (focus on invoice date or issue date)
            - Amounts (ensure you capture the total amount correctly)
            - Line items (capture all individual charges)
            
            Stop and think step by step. Then, extract the invoice data from:
            
            <invoice>
            {document_text}
            </invoice>
    """

    # Use prompt_llm_for_json with our specialized prompt
    return prompt_llm_for_json(action_context=action_context,
                               schema=invoice_schema,
                               prompt=extraction_prompt)


@register_tool(tags=["invoice_processing", "categorization"])
def categorize_expenditure(action_context: ActionContext,
                           description: str) -> str:
    """
    Categorize an invoice expenditure based on a short description.
    
    Args:
        description: A one-sentence summary of the expenditure.
        
    Returns:
        A category name from the predefined set of 20 categories.
    """
    categories = [
        "Office Supplies", "IT Equipment", "Software Licenses",
        "Consulting Services", "Travel Expenses", "Marketing",
        "Training & Development", "Facilities Maintenance", "Utilities",
        "Legal Services", "Insurance", "Medical Services", "Payroll",
        "Research & Development", "Manufacturing Supplies", "Construction",
        "Logistics", "Customer Support", "Security Services", "Miscellaneous"
    ]

    return prompt_expert(
        action_context=action_context,
        description_of_expert=
        "A senior financial analyst with deep expertise in corporate spending categorization.",
        prompt=
        f"Given the following description: '{description}', classify the expense into one of these categories:\n{categories}"
    )


@register_tool(tags=["invoice_processing", "validation"])
def check_purchasing_rules(action_context: ActionContext,
                           invoice_data: dict) -> dict:
    """
    Validate an invoice against company purchasing policies, returning a structured response.
    
    Args:
        invoice_data: Extracted invoice details, including vendor, amount, and line items.
        
    Returns:
        A structured JSON response indicating whether the invoice is compliant and why.
    """
    rules_path = "config/purchasing_rules.txt"

    try:
        with open(rules_path, "r") as f:
            purchasing_rules = f.read()
    except FileNotFoundError:
        purchasing_rules = "No rules available. Assume all invoices are compliant."

    validation_schema = {
        "type": "object",
        "properties": {
            "compliant": {
                "type": "boolean"
            },
            "issues": {
                "type": "string"
            }
        }
    }

    return prompt_llm_for_json(action_context=action_context,
                               schema=validation_schema,
                               prompt=f"""
        Given this invoice data: {invoice_data}, check whether it complies with company purchasing rules.
        The latest purchasing rules are as follows:
        
        {purchasing_rules}
        
        Respond with a JSON object containing:
        - `compliant`: true if the invoice follows all policies, false otherwise.
        - `issues`: A brief explanation of any violations or missing requirements.
        """)


@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Store an invoice in our invoice database. If an invoice with the same number
    already exists, it will be updated.
    
    Args:
        invoice_data: The processed invoice data to store. Make sure this is a 
                        JSON object, not a stringified JSON string. 
                        For example, use {"invoice_number": "1234", ...} instead of 
                        '{"invoice_number": "1234", ...}'

                      The following fields should also be included:
                        - category (str): The expenditure category (e.g., "IT Equipment", "Marketing").
                        - compliant (bool): Whether the invoice passed the compliance check.
                        - issues (list[str], optional): A list of issues found during compliance checks, if any.
        
    Returns:
        A dictionary containing the storage result and invoice number
    """
    # Get our invoice storage from context
    storage = action_context.get("invoice_storage", {})

    # Extract invoice number for reference
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice data must contain an invoice number")

    # Store the invoice
    storage[invoice_number] = invoice_data

    return {
        "status": "success",
        "message": f"Stored invoice {invoice_number}",
        "invoice_number": invoice_number
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


def create_invoice_agent():
    # Create action registry with invoice tools
    action_registry = PythonActionRegistry()

    # Define invoice processing goals
    goals = [
        Goal(
            priority=1,
            name="Persona",
            description=
            "You are an Invoice Processing Agent, specialized in handling invoices efficiently."
        ),
        Goal(priority=1,
             name="Process Invoices",
             description="""
            Your goal is to process invoices accurately. For each invoice:
            1. Extract key details such as vendor, amount, and line items.
            2. Generate a one-sentence summary of the expenditure.
            3. Categorize the expenditure using an expert.
            4. After categorizing the expenditure, validate the invoice against purchasing policies.
            5. Store the processed invoice with categorization and validation status.
            6. Return a summary of the invoice processing results.
            """)
    ]

    # Define agent environment
    environment = PythonEnvironment()

    return Agent(goals=goals,
                 agent_language=AgentFunctionCallingActionLanguage(),
                 action_registry=action_registry,
                 generate_response=generate_response,
                 environment=environment)


def main():
    invoice_text = """
        Invoice #4567
        Date: 2025-02-01
        Vendor: Tech Solutions Inc.
        Items: 
        - Laptop - $1,200
        - External Monitor - $300
        Total: $1,500
    """

    # Create an agent instance
    agent = create_invoice_agent()

    # Process the invoice
    final_memory = agent.run(f"Process this invoice:\n\n{invoice_text}",
                             action_context_props={})

    print('\n')
    print(final_memory.get_memories())


if __name__ == "__main__":
    main()
