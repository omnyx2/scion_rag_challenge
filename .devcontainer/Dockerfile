FROM python:3.11-slim

# 기본 유틸 설치
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# VSCode user 추가
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /workspace

# Python 라이브러리 설치
COPY --chown=vscode:vscode requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt || true

# Jupyter & lint/formatter
RUN pip install jupyter notebook ipykernel ruff
