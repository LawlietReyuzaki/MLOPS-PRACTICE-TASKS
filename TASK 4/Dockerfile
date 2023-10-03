FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install pickle-mixin
EXPOSE 3000
CMD python ./app.py