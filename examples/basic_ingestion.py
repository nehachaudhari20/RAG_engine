from datetime import datetime
from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.ingestion.loaders.json_loader import load_json
from rag_engine.ingestion.loaders.log_loader import load_log

doc1 = load_text("Payment service timeout observed")
doc2 = load_json({"error": "DB_CONN_FAIL", "retry": True})
doc3 = load_log(
    message="Redis connection pool exhausted",
    level="ERROR",
    service="cache-service",
    timestamp=datetime.utcnow()
)

print(doc1)
print(doc2)
print(doc3)
