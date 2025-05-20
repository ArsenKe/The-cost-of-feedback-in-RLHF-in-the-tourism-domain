import json, logging
from firebase_functions import db_fn
from firebase_functions.config import get as config_get
from firebase_admin import credentials, initialize_app, db as rt_db
from google.cloud import pubsub_v1


_initialized = False
def _ensure_admin_sdk_initialized():
    """Initializes Firebase Admin SDK using config from Functions Config if not already done."""
    global _initialized
    if _initialized:
        return

    cfg = config_get()
    admin_config = cfg.get("admin", {})
    admin_creds_json = admin_config.get("credentials")
    db_url = admin_config.get("db_url")

    if not admin_creds_json or not db_url:
        logging.error("Missing 'admin.credentials' or 'admin.db_url' in Functions Config.")
        return 

    initialize_app(
        credentials.Certificate(json.loads(admin_creds_json)),
        {"databaseURL": db_url}
    )
    _initialized = True

REF_PATH = "/feedback/{pushId}"
REGION   = config_get().get("gcp", {}).get("region", "europe-west1")

@db_fn.on_value_created(reference=REF_PATH, region=REGION)
def on_feedback_added(event: db_fn.Event):
    _ensure_admin_sdk_initialized()
    if not _initialized: 
        return {"error": "Admin SDK init failed"}

    cfg = config_get()
    threshold = int(cfg.get("training", {}).get("feedback_threshold", 100))
    project = cfg.get("gcp", {}).get("project")
    topic = cfg.get("pubsub", {}).get("topic")

    total    = len(rt_db.reference("/feedback").get() or {})
    if total < threshold:
        logging.info(f"{total}/{threshold} entries—waiting.")
        return {}

    if not project or not topic:
        logging.error("Missing 'gcp.project' or 'pubsub.topic' in Functions Config. Cannot publish.")
        return {"error": "Missing required configuration for publishing."}

    # Publish message
    publisher = pubsub_v1.PublisherClient()
    tp = publisher.topic_path(project, topic)
    publisher.publish(tp, b"", count=str(total).encode())
    logging.info(f"Published retrain trigger ({total}) to {tp}")
    return {}
