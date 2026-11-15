"""SQLAlchemy engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.utils import PROJECT_ROOT, load_config

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _database_url() -> str:
    cfg = load_config()
    url = str(cfg.get("database_url") or "").strip()
    if url:
        return url
    db_path = PROJECT_ROOT / cfg.get("database_path", "data/okf_metadata.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = _database_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, connect_args=connect_args)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def reset_engine() -> None:
    """Dispose cached engine (tests / DATABASE_PATH changes)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def init_db() -> None:
    """Create all tables if missing."""
    from app.db.models import Base

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
