def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The difference between a and b."""
    return a - b


funcs = [add, subtract]

def dtype_mapper(string: str):
    map = {
        "int": "int",
        "float": "float",
        "str": "string",
        "bool": "boolean",
    }

    for k, v in map.items():
        if k in string:
            return v
        
    return "Object"

def doc_parser(func) -> dict:  # type: ignore
    """ Return a dictionary of parameters and properties. properties include name, type and the description mentioned in the doc.
    if the doc does not have a param, keep it None.
    """

    params = {}
    for param in func.__annotations__.keys():
        if param == "return": continue
        else:
            params[param] = {
                "name": param,
                "type": dtype_mapper(str(func.__annotations__[param])),
                "description": str(func.__doc__).split(f"{param}: ")[1].split("\n")[0],
                "required": [f"{k}: {v}" for k, v in func.__annotations__.items() if k != "return"]
            }

    return params

print(type(add))

print(doc_parser(add))

# for f in funcs:
#     print(f.__name__, f.__doc__, f.__annotations__)