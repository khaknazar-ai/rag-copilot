import os
import socket
import time


def wait_for_tcp(host: str, port: int, name: str, timeout_seconds: int = 90) -> None:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                print(f"{name} is reachable at {host}:{port}")
                return
        except OSError:
            print(f"Waiting for {name} at {host}:{port}...")
            time.sleep(2)

    raise TimeoutError(f"Timed out waiting for {name} at {host}:{port}")


def wait_for_postgres(timeout_seconds: int = 90) -> None:
    import psycopg

    host = os.getenv("POSTGRES_HOST", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "rag_metadata")
    user = os.getenv("POSTGRES_USER", "rag_user")
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError("POSTGRES_PASSWORD must be set to connect to PostgreSQL.")

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with psycopg.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                connect_timeout=3,
            ):
                print("PostgreSQL is ready")
                return
        except Exception as exc:
            print(f"Waiting for PostgreSQL... {exc}")
            time.sleep(2)

    raise TimeoutError("Timed out waiting for PostgreSQL")


def wait_for_redis(timeout_seconds: int = 90) -> None:
    import redis

    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))

    client = redis.Redis(
        host=host,
        port=port,
        db=db,
        socket_connect_timeout=3,
        socket_timeout=3,
        decode_responses=True,
    )

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            if client.ping():
                print("Redis is ready")
                return
        except Exception as exc:
            print(f"Waiting for Redis... {exc}")
            time.sleep(2)

    raise TimeoutError("Timed out waiting for Redis")


def main() -> None:
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    wait_for_tcp(qdrant_host, qdrant_port, "Qdrant")
    wait_for_redis()
    wait_for_postgres()


if __name__ == "__main__":
    main()
