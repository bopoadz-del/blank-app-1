#!/usr/bin/env python3
"""
Edge Processor for Reasoner AI Platform
Runs on Jetson Orin Nano or other edge devices.

Capabilities:
- Offline formula execution
- Local data processing
- Periodic sync with cloud backend
- Sensor data collection
"""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from loguru import logger


class EdgeProcessor:
    """
    Edge processor for offline-first formula execution.
    Syncs with cloud backend when connectivity available.
    """
    
    def __init__(
        self,
        node_id: str,
        backend_url: str = "http://localhost:8000",
        sync_interval: int = 300,
        offline_cache_path: str = "./edge_cache"
    ):
        self.node_id = node_id
        self.backend_url = backend_url
        self.sync_interval = sync_interval
        self.cache_path = Path(offline_cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Local formula cache
        self.formula_cache: Dict[str, Any] = {}
        self.execution_queue = []
        
        logger.info(f"Edge processor initialized: {node_id}")
    
    async def load_formulas_from_backend(self):
        """Download formulas from backend for offline use."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.backend_url}/api/v1/formulas?limit=1000",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    formulas = response.json()
                    
                    for formula in formulas:
                        self.formula_cache[formula["formula_id"]] = formula
                    
                    # Save to disk for offline access
                    cache_file = self.cache_path / "formulas.json"
                    with open(cache_file, "w") as f:
                        json.dump(self.formula_cache, f, indent=2)
                    
                    logger.info(f"Loaded {len(formulas)} formulas from backend")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to load formulas from backend: {e}")
            
            # Try loading from disk cache
            cache_file = self.cache_path / "formulas.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    self.formula_cache = json.load(f)
                logger.info(f"Loaded {len(self.formula_cache)} formulas from cache")
                return True
        
        return False
    
    async def execute_formula_local(
        self,
        formula_id: str,
        input_values: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute formula locally (offline-capable).
        """
        if formula_id not in self.formula_cache:
            return {
                "success": False,
                "error": f"Formula {formula_id} not in local cache"
            }
        
        formula = self.formula_cache[formula_id]
        
        # Simple local execution using eval (in production, use proper SymPy)
        try:
            expression = formula["formula_expression"]
            
            # Replace formula variables with input values
            # This is simplified - production should use SymPy
            result = eval(expression, {"__builtins__": {}}, input_values)
            
            execution_record = {
                "execution_id": f"edge_{self.node_id}_{int(time.time() * 1000)}",
                "formula_id": formula_id,
                "input_values": input_values,
                "output_values": {"result": result},
                "context_data": context or {},
                "status": "completed",
                "execution_time": 0.01,  # Placeholder
                "edge_node_id": self.node_id,
                "timestamp": datetime.utcnow().isoformat(),
                "synced_to_cloud": False
            }
            
            # Queue for sync
            self.execution_queue.append(execution_record)
            
            # Save to disk
            self._save_execution(execution_record)
            
            logger.info(f"Executed formula {formula_id} locally")
            
            return {
                "success": True,
                "result": result,
                "execution_id": execution_record["execution_id"]
            }
            
        except Exception as e:
            logger.error(f"Formula execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_execution(self, execution: Dict[str, Any]):
        """Save execution record to disk."""
        executions_dir = self.cache_path / "executions"
        executions_dir.mkdir(exist_ok=True)
        
        filename = f"{execution['execution_id']}.json"
        with open(executions_dir / filename, "w") as f:
            json.dump(execution, f, indent=2)
    
    async def sync_executions_to_cloud(self):
        """Sync pending executions to cloud backend."""
        if not self.execution_queue:
            return
        
        try:
            async with httpx.AsyncClient() as client:
                for execution in self.execution_queue[:10]:  # Batch of 10
                    try:
                        # Send execution to backend
                        response = await client.post(
                            f"{self.backend_url}/api/v1/formulas/execute",
                            json={
                                "formula_id": execution["formula_id"],
                                "input_values": execution["input_values"],
                                "context_data": execution["context_data"]
                            },
                            timeout=5.0
                        )
                        
                        if response.status_code in [200, 201]:
                            execution["synced_to_cloud"] = True
                            self.execution_queue.remove(execution)
                            logger.info(f"Synced execution {execution['execution_id']}")
                    
                    except Exception as e:
                        logger.error(f"Failed to sync execution: {e}")
                        break  # Stop on first failure
                
        except Exception as e:
            logger.error(f"Sync to cloud failed: {e}")
    
    async def collect_sensor_data(self) -> Dict[str, Any]:
        """
        Collect data from sensors (placeholder).
        In production, integrate with actual sensor APIs.
        """
        # Example sensor data structure
        sensor_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": self.node_id,
            "sensors": {
                "temperature": 25.5,  # °C
                "humidity": 60.0,     # %
                "pressure": 101.3     # kPa
            }
        }
        
        return sensor_data
    
    async def run_sync_loop(self):
        """Main sync loop - runs continuously."""
        logger.info("Starting sync loop")
        
        # Initial formula load
        await self.load_formulas_from_backend()
        
        while True:
            try:
                # Sync executions to cloud
                await self.sync_executions_to_cloud()
                
                # Refresh formulas periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    await self.load_formulas_from_backend()
                
                # Sleep until next sync
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def health_check(self) -> Dict[str, Any]:
        """Check edge processor health."""
        return {
            "node_id": self.node_id,
            "status": "healthy",
            "formulas_cached": len(self.formula_cache),
            "pending_syncs": len(self.execution_queue),
            "timestamp": datetime.utcnow().isoformat()
        }


async def main():
    """Main entry point for edge processor."""
    import sys
    
    # Configuration
    node_id = sys.argv[1] if len(sys.argv) > 1 else "edge_node_1"
    backend_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    # Initialize processor
    processor = EdgeProcessor(
        node_id=node_id,
        backend_url=backend_url,
        sync_interval=300
    )
    
    # Start sync loop
    await processor.run_sync_loop()


if __name__ == "__main__":
    asyncio.run(main())
