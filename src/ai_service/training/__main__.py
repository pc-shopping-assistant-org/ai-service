"""Offline check only: never calls a provider, downloads a model or starts training."""

import argparse
import json
from pathlib import Path

from ai_service.training.dataset import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_dataset(
            args.directory, require_approved=args.require_approved
        )
    except (ValueError, OSError) as error:
        parser.exit(
            1,
            json.dumps(
                {"error": "DATASET_INVALID", "detail": str(error)}, ensure_ascii=False
            )
            + "\n",
        )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
