import time
from datetime import datetime
import random

LOG_FILE = "logs/test.log"

log_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
messages = {
    "INFO": [
        "Service initialized",
        "User login successful",
        "Backup completed",
    ],
    "WARNING": [
        "Memory usage high",
        "Disk space running low",
        "Response time is slow"
    ],
    "ERROR": [
        "Failed to connect to database",
        "Service crashed unexpectedly",
        "Unhandled exception occurred"
    ],
    "DEBUG": [
        "User ID parsed: 12345",
        "Token validated successfully",
        "Request headers logged"
    ]
}

print("Generating logs... (Press Ctrl+C to stop)")

try:
    while True:
        level = random.choice(log_levels)
        message = random.choice(messages[level])
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        log_entry = f"[{level}] {message} - {timestamp}\n"

        with open(LOG_FILE, "a") as f:
            f.write(log_entry)

        print(f"Appended log: {log_entry.strip()}")
        time.sleep(10)

except KeyboardInterrupt:
    print("Stopped log generation.")
