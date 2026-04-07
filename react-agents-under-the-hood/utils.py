from typing import Any, List

def dtype_mapper(string: str):
    map = {
        "int": "integer",   # fixed: OpenAI spec uses "integer" not "int"
        "float": "number",  # fixed: OpenAI spec uses "number" not "float"
        "str": "string",
        "bool": "boolean",
    }

    for k, v in map.items():
        if k in string:
            return v
        
    return "string"  # safe default instead of "Object"


def parse_desc(string: str, param_name: str):  # type: ignore
    before_ = "Returns:"
    after_ = "Args:"

    post = string.split(after_)[1]
    body = post.split(before_)[0]

    param_descs = body.strip().split("\n")

    params = {}
    for param in param_descs:
        if param.strip() == "": continue
        parts = param.split(":", 1)  # fixed: limit split to 1 to avoid breaking on "Literal[...]"
        if len(parts) == 2:
            params[parts[0].strip()] = parts[1].strip()

    for param in params.keys():  # type: ignore
        if param == param_name or param in param_name or param_name in param:
            return params[param]  # type: ignore

    return None


def doc_parser(func) -> dict:  # type: ignore
    params = {}
    for param, annotation in func.__annotations__.items():  # type: ignore
        if param == "return":
            continue
        params[param] = {
            "type": dtype_mapper(str(annotation)),  # type: ignore
            "description": parse_desc(str(func.__doc__), param),  # type: ignore
        }
    return params  # type: ignore


def make_json(funcs: list) -> list:  # type: ignore  # returns a LIST now, not a dict
    tools = []

    for f in funcs:  # type: ignore
        parameters = doc_parser(f)  # type: ignore

        # required = all params that are not Optional
        required: List[Any] = [
            k for k, v in f.__annotations__.items()  # type: ignore
            if k != "return" and "Optional" not in str(v)  # type: ignore
        ]

        tools.append({  # type: ignore
            "type": "function",
            "function": {
                "name": f.__name__,     # type: ignore
                "description": f.__doc__,  # type: ignore
                "parameters": {
                    "type": "object",       # required by OpenAI spec
                    "properties": parameters,
                    "required": required    # just param names, not "name: type"
                }
            }
        })

    return tools  # type: ignore