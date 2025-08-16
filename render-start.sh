#!/usr/bin/env bash

exec uvicorn <your_app_module>:<your_entry_callable> --host 0.0.0.0 --port 8080
