FROM python:3.12.14-trixie

WORKDIR /prod

COPY app /prod/app
COPY requirements.txt /prod/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /prod/requirements.txt

CMD uvicorn app.api.fast:app --host 0.0.0.0 --port $PORT
