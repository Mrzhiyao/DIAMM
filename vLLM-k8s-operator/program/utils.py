import aiofiles
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict
import re

class AsyncTaskLogger:
    _locks: Dict[str, asyncio.Lock] = {}
    
    def __init__(self, log_dir: str = "./async_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    async def log(self, task_id, message: str):
        """异步安全写入方法"""
        # 获取专属锁
        task_id = str(task_id)
        if task_id not in self._locks:
            self._locks[task_id] = asyncio.Lock()
        
        async with self._locks[task_id]:
            await self._write_log(task_id, message)

    async def _write_log(self, task_id, message: str):
        """实际写入操作"""
        task_id = str(task_id)
        safe_id = re.sub(r'[\\/*?:"<>|]', '_', task_id)
        log_file = self.log_dir / f"{safe_id}.log"
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        
        async with aiofiles.open(log_file, mode='a', encoding='utf-8') as f:
            await f.write(log_line)

