from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic plugin tools from a TOML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/synthetic_plugin_generation.toml"),
        help="Path to TOML config copied from conf/synthetic_plugin_generation.example.toml.",
    )
    args = parser.parse_args()

    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    api = cfg.get("api", {})
    generation = cfg.get("generation", {})

    required = {
        "api.base_url": api.get("base_url"),
        "api.api_key": api.get("api_key"),
        "api.model": api.get("model"),
        "generation.real_tool_universe": generation.get("real_tool_universe"),
        "generation.taxonomy": generation.get("taxonomy"),
        "generation.output": generation.get("output"),
        "generation.tool_universe_id": generation.get("tool_universe_id"),
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise SystemExit(f"Missing required config values: {', '.join(missing)}")

    env = dict(os.environ)
    env["TOOL_GEN_API_KEY"] = str(api["api_key"])
    env["TOOL_GEN_BASE_URL"] = str(api["base_url"])

    command = [
        sys.executable,
        "-m",
        "tool_relevance_lab.dataset_generation.run",
        "generate-synthetic-plugins",
        "--real-tool-universe",
        str(generation["real_tool_universe"]),
    ]
    for path in generation.get("existing_tool_universes", []):
        command.extend(["--existing-tool-universe", str(path)])
    if generation.get("creative_brief"):
        command.extend(["--creative-brief", str(generation["creative_brief"])])
    command.extend(
        [
            "--taxonomy",
            str(generation["taxonomy"]),
            "--output",
            str(generation["output"]),
            "--tool-universe-id",
            str(generation["tool_universe_id"]),
            "--model",
            str(api["model"]),
            "--count",
            str(generation.get("count", 40)),
            "--batch-size",
            str(generation.get("batch_size", 10)),
            "--temperature",
            str(generation.get("temperature", 0.8)),
            "--timeout-seconds",
            str(generation.get("timeout_seconds", 120)),
            "--retries",
            str(generation.get("retries", 3)),
        ]
    )
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
