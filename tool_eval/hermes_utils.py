import ast
import json
from typing import List, Optional, Dict, Union
import re


def generate_prompt(sample, enable_system=True):
    prompt = []
    sys_prompt = None

    for turns in sample["conversations"]:
        if turns["from"] == "system":
            if enable_system:
                prompt.append({'content': turns["value"], 'role': 'system'})
            else:
                sys_prompt = turns["value"]
        elif turns["from"] == "human":
            prompt.append({'content': turns["value"], 'role': 'user'})
        elif turns["from"] == "gpt":
            prompt.append({'content': turns["value"], 'role': 'assistant'})
    
    if sys_prompt is not None:
        prompt[0]['content'] = sys_prompt + "\n" + prompt[0]['content']

    return prompt


def clean_json_string(s: str) -> str:
    s = s.replace("\\\\n", "").replace("\\n", "")
    return " ".join(s.split())


def parse_completion(gpt: str) -> Optional[List[Dict[str, Union[str, dict]]]]:
    try:
        if not isinstance(gpt, str):
            raise ValueError(f"Input must be a string, got {type(gpt)}")

        if not gpt.strip():
            return None

        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, gpt, re.DOTALL)

        if not matches:
            return None

        valid_tools = []

        for potential_json in matches:
            try:
                cleaned_json = clean_json_string(potential_json)

                if not cleaned_json:
                    continue

                # First try to parse as regular JSON
                try:
                    tool_call_json = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try converting single quotes to double quotes
                    try:
                        # Use ast.literal_eval to safely evaluate the string as a Python literal
                        tool_call_json = ast.literal_eval(cleaned_json)
                    except (SyntaxError, ValueError) as e:
                        # If both methods fail, try replacing single quotes with double quotes
                        # but only for the outermost quotes
                        if cleaned_json.startswith("'") and cleaned_json.endswith("'"):
                            cleaned_json = cleaned_json[1:-1]  # Remove outer quotes
                            cleaned_json = f'"{cleaned_json}"'  # Add double quotes
                            try:
                                tool_call_json = json.loads(cleaned_json)
                            except:
                                continue
                        else:
                            continue

                if not isinstance(tool_call_json, dict):
                    continue

                restructured_tool = restructure_tool_call(tool_call_json)

                if restructured_tool["name"] is not None:
                    valid_tools.append(restructured_tool)

            except Exception:
                continue

        return valid_tools if valid_tools else None

    except Exception as e:
        return None


def restructure_tool_call(data: Dict) -> Dict:
    result = {"name": None, "arguments": {}}

    def extract_fields(d):
        for k, v in d.items():
            if k == "name" and result["name"] is None:
                result["name"] = v
            elif k == "arguments" and not result["arguments"]:
                result["arguments"] = v
            elif isinstance(v, dict):
                extract_fields(v)

    extract_fields(data)

    if result["name"] is None and "name" in result["arguments"]:
        result["name"] = result["arguments"].pop("name")

    if not result["arguments"]:
        result["arguments"] = {
            k: v for k, v in data.items() if k != "name" and k != "arguments"
        }

    return result


def validate_hermes_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    valid_calls = []

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue

        required_fields = {"name", "arguments"}
        if not all(field in tool_call for field in required_fields):
            continue

        if not isinstance(tool_call["name"], str):
            continue

        if not isinstance(tool_call["arguments"], dict):
            continue

        valid_calls.append(tool_call)

    return valid_calls
