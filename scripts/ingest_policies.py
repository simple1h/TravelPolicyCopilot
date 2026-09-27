from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ingestion import ingest_policies


def main() -> None:
    result = ingest_policies()
    print(
        json.dumps(
            {
                "chunks_count": result.chunks_count,
                "output_path": result.output_path,
                "qdrant_upserted": result.qdrant_upserted,
                "qdrant_error": result.qdrant_error,
                "embedding_provider": result.embedding_provider,
                "embedding_model": result.embedding_model,
                "embedding_dimension": result.embedding_dimension,
                "collection_name": result.collection_name,
                "security_collection_name": result.security_collection_name,
                "embedding_usage": result.embedding_usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
