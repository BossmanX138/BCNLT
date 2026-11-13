@echo off
setlocal
cd /d R:\BossCrafts-Engineering-0.5B
set "PYTHONPATH=R:\BossCrafts-Engineering-0.5B\vendor\tokenizers_runtime_0_23;R:\BossCrafts-Engineering-0.5B"
"C:\Users\bossm\.unsloth\studio\unsloth_studio\Scripts\python.exe" scripts\run_bosscrafts_v5_training_until_complete.py --status artifacts\checkpoints\bosscrafts-engineering-v5-clean\training-progress.json --max-chunks 120
exit /b %ERRORLEVEL%
