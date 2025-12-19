FROM apache/airflow:2.7.1-python3.10

USER airflow

WORKDIR /opt/airflow

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

COPY dags/ ./dags/
COPY plugins/ ./plugins/

EXPOSE 8080
