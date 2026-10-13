"""Explicit native saver bootstrap. Application tables use Alembic separately."""

import asyncio

from ai_service.config.settings import Settings
from ai_service.infrastructure.persistence.runtime import open_stateful_database


async def bootstrap() -> None:
    async with open_stateful_database(Settings()) as (_, saver):
        await saver.setup()


def run() -> None:
    asyncio.run(bootstrap())


if __name__ == "__main__":
    run()
