from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from .types import Message
import json


class HistoryStore(ABC):
    @abstractmethod
    def append(self, message: Message) -> None: ...

    @abstractmethod
    def get(self) -> List[Message]: ...

    @abstractmethod
    def prune(
        self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None
    ) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryHistoryStore(HistoryStore):
    def __init__(self):
        self._messages: List[Message] = []

    def append(self, message: Message) -> None:
        self._messages.append(message)

    def get(self) -> List[Message]:
        return list(self._messages)

    def prune(
        self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None
    ) -> None:
        if max_messages is not None:
            # keep only the last max_messages
            self._messages = self._messages[-max_messages:]
        # Note: token-based pruning requires token estimates; it's left as simple message count here.

    def clear(self) -> None:
        self._messages.clear()


class FileHistoryStore(HistoryStore):
    def __init__(self, path: str):
        self.path = path
        # create file if missing
        open(self.path, "a").close()

    def append(self, message: Message) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "role": message.role,
                        "content": message.content,
                        "meta": message.meta,
                    }
                )
                + "\n"
            )

    def get(self) -> List[Message]:
        out: List[Message] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    out.append(
                        Message(
                            role=d.get("role"),
                            content=d.get("content"),
                            meta=d.get("meta", {}),
                        )
                    )
                except Exception:
                    continue
        return out

    def prune(
        self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None
    ) -> None:
        msgs = self.get()
        if max_messages is not None:
            msgs = msgs[-max_messages:]
        with open(self.path, "w", encoding="utf-8") as f:
            for m in msgs:
                f.write(
                    json.dumps({"role": m.role, "content": m.content, "meta": m.meta})
                    + "\n"
                )

    def clear(self) -> None:
        open(self.path, "w").close()
