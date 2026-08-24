FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PORT=10000
EXPOSE 10000
CMD ["python", "server.py"]
