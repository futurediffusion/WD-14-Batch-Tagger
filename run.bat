@echo off

if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate

pip install -r requirements.txt gradio tqdm requests

python app_gradio.py
pause
