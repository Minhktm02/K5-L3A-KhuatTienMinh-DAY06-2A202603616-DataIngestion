"""Read-only connectivity check for the configured Qdrant server."""
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
url = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333')
api_key = os.getenv('QDRANT_API_KEY') or None
try:
    client = QdrantClient(url=url, api_key=api_key, timeout=5)
    info = client.info()
    collections = client.get_collections().collections
    print(f'Qdrant: {info.version} | URL: {url} | Collections: {len(collections)}')
    client.close()
except Exception as exc:
    sys.exit(f'Qdrant connection failed for {url}: {type(exc).__name__}: {exc}')
print('Qdrant setup OK.')
