# Demo
This is a WEB application demo that allows the user interaction to the model. This demo requires the API running locally either in a docker container built from `docker-compose.yaml` or on local computer.

***This application was not made for comercial purposes.**

---
### Run
#### Docker
```
$ docker compose up
```
#### Local
```
# API
$ uvicorn app.main:app --port 8000 --host 0.0.0.0
# Demo
$ chmod +x demo/run.sh && ./demo/run.sh
```
---