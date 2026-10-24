FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROTEIN_DATA_DIR=/app/data

RUN useradd --create-home appuser
WORKDIR /app

COPY pyproject.toml README.md ./
COPY protein ./protein
RUN pip install --no-cache-dir .

COPY data ./data
RUN chown -R appuser:appuser /app/data

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=4).raise_for_status()"

CMD ["uvicorn", "protein.api:app", "--host", "0.0.0.0", "--port", "8000"]
