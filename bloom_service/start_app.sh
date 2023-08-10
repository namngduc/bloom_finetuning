#!/bin/bash

gunicorn main:app -c gunicorn.py --worker-class uvicorn.workers.UvicornWorker