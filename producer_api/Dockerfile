FROM python:3.10-slim

WORKDIR /app

RUN pip install fastapi uvicorn faker

COPY app.py .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
