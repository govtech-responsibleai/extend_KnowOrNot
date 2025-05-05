from typing import Optional


class InitialCallFailedException(Exception):
    def __init__(self, model_name: str, error_message: Optional[str] = None):
        super().__init__(
            f"Unable to use model {model_name} with error: {error_message if error_message else 'No error message provided'}"
        )
