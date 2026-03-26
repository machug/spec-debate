"""Session state management and checkpointing for adversarial spec debates."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path.home() / ".config" / "adversarial-spec" / "sessions"
CHECKPOINTS_DIR = Path.cwd() / ".adversarial-spec-checkpoints"


@dataclass
class SessionState:
    """Persisted state for resume functionality."""

    session_id: str
    spec: str
    round: int
    doc_type: str
    models: list
    focus: Optional[str] = None
    persona: Optional[str] = None
    preserve_intent: bool = False
    created_at: str = ""
    updated_at: str = ""
    history: list = field(default_factory=list)

    def save(self):
        """Save session state to disk."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        path = SESSIONS_DIR / f"{self.session_id}.json"
        if not path.resolve().is_relative_to(SESSIONS_DIR.resolve()):
            raise ValueError(f"Invalid session ID: {self.session_id}")
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, session_id: str) -> "SessionState":
        """Load session state from disk."""
        path = SESSIONS_DIR / f"{session_id}.json"
        if not path.resolve().is_relative_to(SESSIONS_DIR.resolve()):
            raise ValueError(f"Invalid session ID: {session_id}")
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")
        data = json.loads(path.read_text())
        return cls(**data)

    @classmethod
    def list_sessions(cls) -> list[dict]:
        """List all saved sessions."""
        if not SESSIONS_DIR.exists():
            return []
        sessions = []
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                sessions.append(
                    {
                        "id": data["session_id"],
                        "round": data["round"],
                        "doc_type": data["doc_type"],
                        "updated_at": data.get("updated_at", ""),
                    }
                )
            except Exception:
                pass
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)


def save_checkpoint(spec: str, round_num: int, session_id: Optional[str] = None):
    """Save spec checkpoint for this round."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{session_id}-" if session_id else ""
    path = CHECKPOINTS_DIR / f"{prefix}round-{round_num}.md"
    if not path.resolve().is_relative_to(CHECKPOINTS_DIR.resolve()):
        raise ValueError(f"Invalid session ID: {session_id}")
    path.write_text(spec)
    print(f"Checkpoint saved: {path}", file=sys.stderr)
