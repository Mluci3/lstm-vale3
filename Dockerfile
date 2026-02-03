FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PIP_ONLY_BINARY=:all:

RUN pip install --upgrade pip
RUN pip install --only-binary=:all: numpy==1.26.4
RUN pip install --only-binary=:all: scipy==1.11.4
RUN pip install --only-binary=:all: h5py==3.11.0
RUN pip install --only-binary=:all: tensorflow==2.18.0
RUN pip install --only-binary=:all: scikit-learn==1.3.2
RUN pip install --only-binary=:all: joblib==1.3.2
RUN pip install --only-binary=:all: pandas==2.1.4
RUN pip install --only-binary=:all: fastapi==0.109.0
RUN pip install --only-binary=:all: mangum==0.17.0
RUN pip install --only-binary=:all: pydantic==2.5.3
RUN pip install yfinance==0.2.40

COPY src/api/ ${LAMBDA_TASK_ROOT}/src/api/
COPY src/__init__.py ${LAMBDA_TASK_ROOT}/src/

COPY models/lstm_vale3.h5 ${LAMBDA_TASK_ROOT}/models/
COPY models/scaler.joblib ${LAMBDA_TASK_ROOT}/models/
COPY models/metricas.json ${LAMBDA_TASK_ROOT}/models/

RUN mkdir -p ${LAMBDA_TASK_ROOT}/data/processed
COPY data/processed/config.json ${LAMBDA_TASK_ROOT}/data/processed/

CMD ["src.api.main.handler"]