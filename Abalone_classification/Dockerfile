FROM python:3.8-slim
COPY . /Abalone_classification/deploy
COPY  . /requirements.txt/Abalone_classification/deploy
WORKDIR /Abalone_classification/deploy
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT [ "python" ]
CMD ["main.py"]