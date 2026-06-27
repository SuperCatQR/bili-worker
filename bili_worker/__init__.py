"""bili_worker — GPL-3.0 worker process for F2 bilibili-api isolation.

This package is the **only** place that imports ``bilibili_api``. It is an
independently distributable GPL-3.0 component (own ``pyproject.toml`` + ``LICENSE``);
the main ``bili_unit`` process never imports or links it — it spawns it as a
subprocess and talks an arm's-length stdio JSON protocol (see
``bili_unit/docs/ipc-contract-f2.md``).

Stage 2 status: this module currently provides the protocol-codec and error-mapping
foundation (Step 2). The op dispatch loop and SDK callable catalog land in later steps.
"""

from __future__ import annotations

__version__ = "0.1.0"
PROTOCOL_VERSION = "1.0"

__all__ = ["PROTOCOL_VERSION", "__version__"]
