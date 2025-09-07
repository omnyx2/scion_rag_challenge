class LLMClientError(Exception):
    pass


class UnsupportedFeatureError(LLMClientError):
    """Raised when a backend doesn't support a requested feature (e.g. image input)."""

    pass


class AuthError(LLMClientError):
    pass


class APIError(LLMClientError):
    pass
