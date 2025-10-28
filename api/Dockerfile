FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app /app/app
COPY config /app/config
RUN mkdir -p /app/data /app/models /app/data/sessions
ENV CONFIG_PATH=/app/config/server_config.yaml SAVE_LOGS=true REDIS_URL=redis://redis:6379/0
EXPOSE 8080
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]