#!/bin/bash

PROJECT_ID="log-anomaly-detector"
LOG_NAME="custom_error_test"

# Sample log messages
logs=(
  "INFO: System boot completed successfully."
  "WARNING: Memory usage at 85%."
  "ERROR: Simulated crash on instance-200"
  "INFO: Background job finished."
  "ERROR: Disk quota exceeded on instance-7"
  "DEBUG: Cache miss for user profile lookup"
  "ERROR: Timeout while connecting to DB"
)

for log in "${logs[@]}"; do
  if [[ $log == ERROR* ]]; then
    severity="ERROR"
  elif [[ $log == WARNING* ]]; then
    severity="WARNING"
  elif [[ $log == INFO* ]]; then
    severity="INFO"
  elif [[ $log == DEBUG* ]]; then
    severity="DEBUG"
  else
    severity="DEFAULT"
  fi

  echo "Sending log: [$severity] $log"
  gcloud logging write "$LOG_NAME" "$log" --severity="$severity" --project="$PROJECT_ID"
  sleep 2  
done
