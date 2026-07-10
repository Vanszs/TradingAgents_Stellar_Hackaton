"""Entry point for news aggregator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from news_aggregator.aggregator import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
