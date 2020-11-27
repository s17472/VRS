FROM python:3.8-slim-buster

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential

RUN pip install --upgrade pip setuptools poetry

WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

COPY . .

ENTRYPOINT [ "./entrypoint.sh" ]