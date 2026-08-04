from __future__ import annotations


class ExecutionContext:
    def __init__(self, initial: dict | None = None) -> None:
        self.variables: dict = dict(initial or {})

    def set(self, key: str, value: object) -> None:
        self.variables[key] = value

    def get(self, key: str) -> object | None:
        return self.variables.get(key)

    def all(self) -> dict:
        return dict(self.variables)
