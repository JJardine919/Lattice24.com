FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PORT=10000
# server.py binds 127.0.0.1 by default so a run on a laptop does not publish an
# unauthenticated engine endpoint to the LAN. A container is the one place a
# public bind is correct: Render routes to it from outside, and a loopback bind
# here is simply unreachable.
ENV HOST=0.0.0.0
# ...and because Render's proxy is then the only peer this socket ever sees,
# the per-IP throttle must read the forwarded client instead. Without this the
# whole internet shares one bucket.
ENV TRUST_PROXY=1
EXPOSE 10000
CMD ["python", "server.py"]
