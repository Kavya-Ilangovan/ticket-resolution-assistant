"""Key/value abstraction: Redis in prod (cache + rate limit), in-process dict otherwise. Fails open."""
from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

log = logging.getLogger(__name__)


class KV(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int) -> None: ...
    def incr_window(self, key: str, window_s: int) -> int: ...
    def ping(self) -> bool: ...
    def queue_depth(self, name: str) -> int | None: ...


class MemoryKV:
    def __init__(self):
        self._d: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if not item:
                return None
            exp, val = item
            if exp < time.time():
                self._d.pop(key, None)
                return None
            return val

    def set(self, key, value, ttl):
        with self._lock:
            if len(self._d) > 50_000:  # crude bound; Redis handles eviction properly in prod
                self._d.clear()
            self._d[key] = (time.time() + ttl, value)

    def incr_window(self, key, window_s):
        bucket = f"{key}:{int(time.time() // window_s)}"
        with self._lock:
            exp, val = self._d.get(bucket, (time.time() + window_s, "0"))
            n = int(val) + 1
            self._d[bucket] = (exp, str(n))
            return n

    def ping(self):
        return True

    def queue_depth(self, name):
        return None


class RedisKV:
    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.25, socket_connect_timeout=0.5)

    def get(self, key):
        try:
            return self.r.get(key)
        except Exception as e:  # cache must never take the API down
            log.warning("redis get failed: %s", e)
            return None

    def set(self, key, value, ttl):
        try:
            self.r.set(key, value, ex=ttl)
        except Exception as e:
            log.warning("redis set failed: %s", e)

    def incr_window(self, key, window_s):
        bucket = f"{key}:{int(time.time() // window_s)}"
        try:
            pipe = self.r.pipeline()
            pipe.incr(bucket)
            pipe.expire(bucket, window_s + 1)
            return int(pipe.execute()[0])
        except Exception as e:
            log.warning("redis rate-limit failed open: %s", e)
            return 0

    def ping(self):
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def queue_depth(self, name):
        try:
            return int(self.r.llen(name))
        except Exception:
            return None


def build_kv(redis_url: str | None) -> KV:
    return RedisKV(redis_url) if redis_url else MemoryKV()
