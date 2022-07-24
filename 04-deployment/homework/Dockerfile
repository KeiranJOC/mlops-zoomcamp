FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

RUN mkdir output

ENTRYPOINT ["python", "starter.py"]
