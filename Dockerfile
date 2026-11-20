# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.14.10
FROM python:${PYTHON_VERSION}-slim AS base

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1

WORKDIR app

COPY flask-app/requirements.txt app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r app/requirements.txt

COPY flask-app/ app/

COPY model/vectorizer.pkl model/vectorizer.pkl

EXPOSE 5000

CMD ["python", "app/app.py"]