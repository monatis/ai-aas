# ai-aas
AI as a Service. Production is all you need

## Acknowledgement
Google supported this work by providing Google Cloud credit. Thank you Google for supporting the open source! 🎉

## Overview
In this project, I aim at developing an embarrassingly simple, production ready AI as a Service.

## Motivation
There are lots of free and open AI/ML projects published by great folks across the globe. However, it is not that easy to integrate such models into existing projects and deploy them in production. Therefore, we need a simple way to run them as an API service on various deployment targets.

## Design Goals
Ideally, this project should be:
- technically and legally ready for use in production.
- containerized and self-contained.
- configurable with simple config files and/or environment variables.
- cross-platform and accelerator-agnostic.
- deployable with only a few clicks and/or commands.
- scalable for extreme use cases.
- modular so that it can support choosing different sets of models.
- predictable and self-documented.
- maintainable with minimum dependencies.
- easily expandable with new models.
- usable for batch and/or online prediction.

## How to use
This project makes use of [Docker Compose Profiles](https://docs.docker.com/compose/profiles/) to support optional enablement of services, so you need to have Docker engine V20.10.5 (or above) installed.

```bash
git clone https://github.com/monatis/ai-aas.git
cd ai-aas
export COMPOSE_PROFILES=zsl,ner
docker-compose up -d
```

## Current supported tasks
- Zero-shot text classification
- Named entity recognition
- Question answering
- Question paraphrasing
- Question generation (?)
- Abstractive summarization

**Very soon**
- Text clustering
- Scalable semantic search
- And more

## Plan of Attack
- Identify a few suitable models for the initial release.
- Identify the minimum dependencies (TensorFlow, FastAPI, Redis, Traefik and possibly a few more).
- Design the infrastructure.
- Implement abstract request and response layers.
- Implement dependable request and response schemas for different data modalities.
- Implement unified and reusable preprocessing components.
- Implement prediction workers.
- Use other great services such as TF Hub as much as possible.
- Make it configurable thanks to docker-compose profiles.
- Make it optimized.
- Power great projects that need production-grade AI/ML services!
