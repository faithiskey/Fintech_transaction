FROM python:3.13.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /main

COPY requirements.txt /main/
RUN  pip install --no-cache-dir -r requirements.txt


COPY main/ ./main/
COPY model/ ./model/
COPY data/ ./data/

RUN useradd --create-home appuser && chown -R appuser:appuser /main
USER appuser


EXPOSE 8000

CMD ["bash", "-c", "uvicorn main.app:app --host 0.0.0.0 --port 8000 & streamlit run main/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"]