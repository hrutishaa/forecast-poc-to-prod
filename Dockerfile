FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md config.toml ./
COPY forecast ./forecast
RUN pip install --no-cache-dir . && python -m forecast.train
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
