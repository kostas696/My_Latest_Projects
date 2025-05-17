from elasticsearch import Elasticsearch
import requests
import time
import datetime

# Constants
ES_HOST = "http://localhost:9200"
ES_INDEX = "filebeat-*"
OLLAMA_API = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"
POLL_INTERVAL = 10  # seconds
QUERY_SIZE = 5
DEBUG = True

# Initialize Elasticsearch client with correct Accept headers for version 8
es = Elasticsearch(
    ES_HOST,
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# Initialize timestamp window
last_timestamp = "now-1m"

def explain_log_with_llm(message):
    """Send a log message to the local LLM for summarization and resolution."""
    payload = {
        "model": LLM_MODEL,
        "prompt": f"Summarize and suggest resolution steps for this log error:\n{message}",
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_API, json=payload)
        if res.ok:
            return res.json().get("response", "[No response from LLM]")
        return f"[LLM API error] Status Code: {res.status_code}"
    except Exception as e:
        return f"[LLM exception] {e}"

def format_ts(ts):
    """Format ISO timestamp for printing."""
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ts

print("Starting log tailing and LLM analysis...")

while True:
    try:
        query = {
            "size": QUERY_SIZE,
            "sort": [{"@timestamp": {"order": "asc"}}],
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gt": last_timestamp}}},
                        {"match_phrase": {"log.file.path": "/logs/test.log"}}
                    ],
                    "should": [
                        {"match": {"message": "ERROR"}}
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        res = es.search(index=ES_INDEX, body=query)
        hits = res.get("hits", {}).get("hits", [])

        if hits:
            for hit in hits:
                src = hit["_source"]
                message = src.get("message", "")
                ts = src.get("@timestamp", "")
                last_timestamp = ts

                print(f"\nNew log at {format_ts(ts)}:\n{message}")
                explanation = explain_log_with_llm(message)
                print(f"LLM Response:\n{explanation}")
        else:
            if DEBUG:
                print("No new logs.")
        time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nExiting on user interrupt.")
        break
    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(POLL_INTERVAL)
