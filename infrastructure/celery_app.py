# ===========================================
# Alpha Sentiment Engine — Celery Configuration
# ===========================================
# This file sets up Celery — the "kitchen manager" that
# assigns headlines to workers.
#
# It tells Celery:
#   - Where's the queue? (Redis at localhost:6379)
#   - What format to use? (JSON — human readable)
#   - Where to find tasks? (infrastructure/tasks.py)
#
# This file is just SETTINGS. The actual work happens in tasks.py.
# ===========================================

from celery import Celery

# Create the Celery app (give it a name)
app: Celery = Celery("alpha_sentiment")

# Tell Celery where Redis is and how to communicate
app.conf.update(
    # Where to find the queue (Redis database 0)
    broker_url="redis://localhost:6379/0",

    # Where to store results (Redis database 1)
    result_backend="redis://localhost:6379/1",

    # Send everything as JSON (human-readable, easy to debug)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# Tell Celery to look for tasks in the "infrastructure" folder
app.autodiscover_tasks(["infrastructure"])
