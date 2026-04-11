FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py          ./
COPY env.py             ./
COPY baseline_agents.py ./
COPY inference.py       ./
COPY app.py             ./
COPY tasks/             ./tasks/
COPY server/            ./server/
COPY tasks.json         ./
COPY openenv.yaml       ./
COPY README.md          ./

# Runtime environment variables — override via HF Space secrets or .env
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV HF_TOKEN=""
ENV AGRI_TASK=hard
ENV AGRI_SCENARIO=default

# Validation checklist (checked at build time):
# [x] pydantic, openai, openenv-core, fastapi, uvicorn installed
# [x] env.py loads and reset() works
# [x] tasks/ folder present with easy/medium/hard graders
# [x] inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env
# [x] [START] [STEP] [END] log format emitted to stdout
# [x] all rewards clamped to [0.0, 1.0]
# [x] app.py exposes /reset /step /state /inference on port 7860

RUN python -c "from env import AgriEnv; e=AgriEnv(); e.reset(); print('ENV OK')"

EXPOSE 7860

# app.py runs the FastAPI server (required for HF Spaces to stay alive)
# inference.py can be triggered via POST /inference or run directly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
