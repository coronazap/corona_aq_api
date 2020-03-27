FROM python:3.6
RUN pip3 install -r requirements.txt
CMD cd app && python app.py     