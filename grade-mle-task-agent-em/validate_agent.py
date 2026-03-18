#!/usr/bin/env python3
"""
Validate custom_agent.py implementation against base_agent.py contract.

Checks:
1. No duplicate method names (except required abstract methods)
2. Required methods have correct signatures
"""

import ast
import sys
from typing import Dict, List, Tuple


def get_class_methods(filepath: str, class_name: str) -> Dict[str, ast.FunctionDef]:
    """Extract all methods from a class."""
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)

    methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[item.name] = item
    return methods


def get_method_signature(func: ast.FunctionDef) -> Tuple[List[str], List[str]]:
    """Extract method signature (args, return type)."""
    args = []
    for arg in func.args.args:
        arg_name = arg.arg
        # Get type annotation if present
        if arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                arg_type = arg.annotation.id
            elif isinstance(arg.annotation, ast.Constant):
                arg_type = str(arg.annotation.value)
            else:
                arg_type = ast.unparse(arg.annotation)
            args.append(f"{arg_name}: {arg_type}")
        else:
            args.append(arg_name)

    # Check if async
    is_async = isinstance(func, ast.AsyncFunctionDef)

    return args, is_async


def validate_agent(
    base_agent_path: str, custom_agent_path: str
) -> Tuple[bool, List[str]]:
    """
    Validate custom agent against base agent.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    try:
        # Get base agent methods
        base_methods = get_class_methods(base_agent_path, "BaseAgent")
        if not base_methods:
            errors.append("Could not find BaseAgent class in base_agent.py")
            return False, errors

        # Find the custom agent class (any class inheriting from BaseAgent)
        with open(custom_agent_path, "r") as f:
            tree = ast.parse(f.read(), filename=custom_agent_path)

        custom_class_name = None
        custom_methods = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from BaseAgent
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseAgent":
                        custom_class_name = node.name
                        for item in node.body:
                            if isinstance(
                                item, (ast.FunctionDef, ast.AsyncFunctionDef)
                            ):
                                custom_methods[item.name] = item
                        break

        if not custom_class_name:
            errors.append(
                "Could not find a class inheriting from BaseAgent in custom_agent.py"
            )
            return False, errors

        # Required methods that must be implemented
        required_methods = {"start", "continue_agent", "abort"}

        # Methods that should NOT be overridden (except required ones)
        protected_methods = set()
        for method_name in base_methods:
            if method_name not in required_methods and not method_name.startswith("_"):
                # Public helper methods from BaseAgent
                protected_methods.add(method_name)

        # Check 1: Required methods are implemented
        for required in required_methods:
            if required not in custom_methods:
                errors.append(
                    f"Required method '{required}' is not implemented in {custom_class_name}"
                )

        # Check 2: Required methods have correct signatures
        signature_checks = {
            "start": (["self"], True),  # (args, is_async)
            "continue_agent": (
                ["self", "user_message", "new_max_steps", "step_number", "branch_name"],
                True,
            ),
            "abort": (["self"], True),
        }

        for method_name, (expected_args, expected_async) in signature_checks.items():
            if method_name in custom_methods:
                custom_func = custom_methods[method_name]
                actual_args, actual_async = get_method_signature(custom_func)

                # Check async
                if actual_async != expected_async:
                    if expected_async:
                        errors.append(
                            f"Method '{method_name}' must be async (use 'async def')"
                        )
                    else:
                        errors.append(f"Method '{method_name}' should not be async")

                # Check args (only check required args, allow extra optional ones)
                actual_arg_names = [arg.split(":")[0].strip() for arg in actual_args]
                for i, expected_arg in enumerate(expected_args):
                    if i >= len(actual_arg_names):
                        errors.append(
                            f"Method '{method_name}' missing required parameter: {expected_arg}"
                        )
                    elif actual_arg_names[i] != expected_arg:
                        errors.append(
                            f"Method '{method_name}' parameter {i + 1} should be '{expected_arg}', got '{actual_arg_names[i]}'"
                        )

        # Check 3: No overriding of protected helper methods
        for method_name in custom_methods:
            if method_name in protected_methods:
                errors.append(
                    f"Method '{method_name}' from BaseAgent should not be overridden. "
                    f"This is a helper method provided by the base class."
                )

        return len(errors) == 0, errors

    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate_agent.py <base_agent.py> <custom_agent.py>")
        sys.exit(1)

    base_agent_path = sys.argv[1]
    custom_agent_path = sys.argv[2]

    is_valid, errors = validate_agent(base_agent_path, custom_agent_path)

    if is_valid:
        print("✓ Validation passed")
        sys.exit(0)
    else:
        print("✗ Validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
