# Edge Processor

Runs on Jetson Orin Nano or other edge devices for offline-first formula execution.

## Setup

```bash
pip install -r requirements.txt
python edge_processor.py edge_node_1 http://backend:8000
```

## Features

- Offline formula execution
- Local caching
- Periodic cloud sync
- Sensor data collection (extensible)
