import os
import json
import logging

from firebase_functions import db_fn, pubsub_fn
from firebase_functions.config import get as config_get
from firebase_admin import credentials, initialize_app, db as rt_db
from google.cloud import pubsub_v1, aiplatform
from firebase_functions.pubsub_fn import CloudEvent

_initialized = False

def _ensure_admin():
    global _initialized
    if _initialized:
        return

    cfg = config_get()
    admin_cfg = cfg.get("admin", {})
    b64_creds = admin_cfg.get("credentials_b64")
    db_url    = admin_cfg.get("db_url")
    if not b64_creds or not db_url:
        logging.error("Missing admin.credentials_b64 or admin.db_url in Functions Config")
        return

    svc_json = json.loads(
        base64.b64decode(b64_creds).decode("utf-8")
    )
    initialize_app(
        credentials.Certificate(svc_json),
        {"databaseURL": db_url}
    )
    _initialized = True

REF_PATH = "/feedback/{pushId}"
REGION   = config_get().get("gcp", {}).get("region", "europe-west1")

@db_fn.on_value_created(reference=REF_PATH, region=REGION)
def on_feedback_added(event: db_fn.Event):
    _ensure_admin()
    if not _initialized:
        return {"error": "Admin init failed"}

    cfg       = config_get()
    threshold = int(cfg.get("training", {}).get("feedback_threshold", 100))
    total     = len(rt_db.reference("/feedback").get() or {})
    if total < threshold:
        logging.info(f"{total}/{threshold} entries—waiting.")
        return {}

    project = cfg.get("gcp", {}).get("project")
    topic   = cfg.get("pubsub", {}).get("topic")
    if not project or not topic:
        logging.error("Missing gcp.project or pubsub.topic")
        return {}

    publisher = pubsub_v1.PublisherClient()
    tp = publisher.topic_path(project, topic)
    publisher.publish(tp, b"", count=str(total).encode())
    logging.info(f"Published retrain trigger ({total}) → {tp}")
    return {}

TOPIC = config_get().get("pubsub", {}).get("topic", "retrain-dpo")
REGION = config_get().get("gcp", {}).get("region", "europe-west1")

@pubsub_fn.on_message_published(topic=TOPIC, region=REGION)
def launch_dpo(data, context):
    event = CloudEvent(data, context)

    cfg     = config_get()
    project = cfg.get("gcp", {}).get("project")
    image   = cfg.get("training", {}).get("image")
    if not project or not image:
        logging.error("Missing gcp.project or training.image")
        return {}

    aiplatform.init(project=project, location=REGION)
    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"dpo_retrain_{context.id}",
        container_uri=image,
    )
    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        environment_variables={
            "HF_TOKEN":        cfg.get("hf", {}).get("token"),
            "HF_ADAPTER_REPO": cfg.get("hf", {}).get("adapter_repo"),
            "BASE_MODEL":      cfg.get("training", {}).get("base_model", "google/mt5-large"),
            "DPO_BETA":        cfg.get("training", {}).get("dpo_beta", "1e-3"),
        }
    )
    logging.info(f"Submitted Vertex AI job {job.display_name}")
    return {}
