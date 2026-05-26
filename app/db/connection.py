"""Пул соединений к PostgreSQL."""
from contextlib import contextmanager
from psycopg_pool import ConnectionPool
from app.config import settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=settings.dsn, min_size=1, max_size=4, open=True)
    return _pool


@contextmanager
def get_conn():
    pool = get_pool()
    with pool.connection() as conn:
        yield conn