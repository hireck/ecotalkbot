# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.9.6
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN apt-get update -q \
  && apt-get install --no-install-recommends -y gcc zlib1g zlib1g-dev libffi-dev
RUN pip install --upgrade pip setuptools wheel
RUN pip install setuptools --upgrade
RUN pip install transformers==4.47.1
RUN pip install transformers FlagEmbedding==1.3.4
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt --use-deprecated=legacy-resolver

# Switch to the non-privileged user to run the application.
#USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 80

# Run the application.
#CMD streamlit run ecotalkbot/ecotalkbot.py --server.port 8501
CMD streamlit run ecotalkbot.py --server.port 80
