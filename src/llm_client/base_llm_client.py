class BaseLLMClient(ABC):
    def __init__(self, model: str, user_api_key: str | None = None, max_retries: int = 3, timeout_s: float = 30.0, history_store: HistoryStore | None = None): ...

    def set_mode(self, mode: Literal["instruct", "uninstruct"]) -> None: ...
    def add_system(self, content: str) -> None: ...
    def add_user(self, content: str, images: list[ImageInput] | None = None) -> None: ...
    def add_assistant(self, content: str) -> None: ...
    def clear_history(self) -> None: ...

    def generate(self, temperature: float = 0.7, max_tokens: int | None = None) -> GenerationResult: ...
    async def agenerate(self, temperature: float = 0.7, max_tokens: int | None = None) -> GenerationResult: ...

@dataclass
class ImageInput:
    url: str | None = None
    bytes: bytes | None = None
    mime_type: str | None = None

@dataclass
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    raw: dict  # 원시 응답


