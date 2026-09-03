import os
from enum import Enum

class DiagnosticCategory(str, Enum):
    input_sources = "input_sources"
    prompts = "prompts"
    model = "model"
    output_schema = "output_schema"
    raw_response = "raw_response"
    structured_output = "structured_output"
    execution = "execution"

ALL_CATEGORIES = frozenset([c.value for c in DiagnosticCategory])

def parse_diagnostic_config(value: str | None, default: str) -> frozenset[str]:
    if value is None:
        value = default

    if not value.strip():
        raise ValueError("Diagnostic configuration cannot be empty.")

    tokens = [t.strip().lower() for t in value.split(",")]
    
    if len(tokens) == 1 and tokens[0] == "none":
        return frozenset()
    
    if len(tokens) == 1 and tokens[0] == "all":
        return ALL_CATEGORIES
        
    categories = set()
    for token in tokens:
        if not token:
            raise ValueError("Diagnostic configuration cannot contain empty tokens.")
        if token in ["all", "none"]:
            raise ValueError(f"Diagnostic configuration cannot mix '{token}' with other categories.")
        if token not in ALL_CATEGORIES:
            raise ValueError(f"Unknown diagnostic category: '{token}'")
        if token in categories:
            raise ValueError(f"Duplicate diagnostic category: '{token}'")
        categories.add(token)
        
    return frozenset(categories)

def get_role_policies(app_config=None) -> dict[str, frozenset[str]]:
    """Get policies, optionally using Flask config or falling back to env/defaults for testing."""
    if app_config is None:
        return {
            "superadmin": parse_diagnostic_config(os.getenv("DIAGNOSTICS_SUPERADMIN"), "all"),
            "admin": parse_diagnostic_config(os.getenv("DIAGNOSTICS_ADMIN"), "none"),
            "user": parse_diagnostic_config(os.getenv("DIAGNOSTICS_USER"), "none")
        }
    else:
        return {
            "superadmin": app_config.get("DIAGNOSTICS_SUPERADMIN", ALL_CATEGORIES),
            "admin": app_config.get("DIAGNOSTICS_ADMIN", frozenset()),
            "user": app_config.get("DIAGNOSTICS_USER", frozenset())
        }

def diagnostic_policy_for(user) -> frozenset[str]:
    from flask import current_app
    
    if not user or not user.role or not user.role.name:
        return frozenset()
        
    role_name = user.role.name.lower()
    
    # Fast path if we're in a Flask app context
    if current_app:
        policies = {
            "superadmin": current_app.config.get("DIAGNOSTICS_SUPERADMIN", ALL_CATEGORIES),
            "admin": current_app.config.get("DIAGNOSTICS_ADMIN", frozenset()),
            "user": current_app.config.get("DIAGNOSTICS_USER", frozenset())
        }
    else:
        policies = get_role_policies()
        
    return policies.get(role_name, frozenset())
