import logging
from firebase_functions import pubsub_fn
from firebase_functions.config import get as config_get
from google.cloud import aiplatform
from firebase_functions.pubsub_fn import CloudEvent


cfg0     = config_get()
TOPIC    = cfg0["pubsub"]["topic"]
REGION   = cfg0["gcp"].get("region", "europe-west1")

@pubsub_fn.on_message_published(topic=TOPIC, region=REGION)
def launch_dpo(event: pubsub_fn.CloudEvent):
    cfg   = config_get()
    project = cfg["gcp"]["project"]
    image   = cfg["training"]["image"]
    if not project or not image:
        logging.error("Missing gcp.project or training.image")
        return {}

    aiplatform.init(project=project, location=REGION)
    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"dpo_retrain_{event.id}",
        container_uri=image,
    )
    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        environment_variables={
            "HF_TOKEN":        cfg["hf"]["token"],
            "HF_ADAPTER_REPO": cfg["hf"]["adapter_repo"],
            "BASE_MODEL":      cfg["training"]["base_model"],
            "DPO_BETA":        cfg["training"]["dpo_beta"],
        },
    )
    logging.info(f"Submitted job {job.display_name}")
    return {}
