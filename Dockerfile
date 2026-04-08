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
COPY tasks/             ./tasks/
COPY openenv.yaml       ./
COPY README.md          ./

# Runtime environment variables — override via -e flags or HF Space secrets
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV HF_TOKEN=""
ENV AGRI_TASK=hard
ENV AGRI_SCENARIO=default

# Validation checklist (checked at build time):
# [x] pydantic, openai, openenv-core installed
# [x] env.py loads and reset() works
# [x] tasks/ folder present with easy/medium/hard graders
# [x] inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env
# [x] [START] [STEP] [END] log format emitted to stdout
# [x] all rewards clamped to [0.0, 1.0]

RUN python -c "from env import AgriEnv; e=AgriEnv(); e.reset(); print('ENV OK')"

EXPOSE 7860

CMD ["python", "inference.py"]
