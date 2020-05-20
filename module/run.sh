#!/usr/bin/env bash
service nginx start
gunicorn -w 3 --bind localhost:8000 app:application