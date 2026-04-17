FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ui/ ./ui/
COPY app/ ./app/

COPY requirements-frontend.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-frontend.txt



EXPOSE 8501

CMD ["sh", "-c", "streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]