import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import os


# Explicitly point to your service account key file
SERVICE_ACCOUNT_KEY_PATH = r"C:\Users\arsen\OneDrive\Desktop\rlhf-feedback-app\serviceAccountKey.json"
DATABASE_URL = "https://rlhf-2e2cc-default-rtdb.europe-west1.firebasedatabase.app" 

try:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
         cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    else:
         cred = credentials.ApplicationDefault()

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': DATABASE_URL
        })
    print("Firebase Admin SDK initialized successfully.")

except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    print("Ensure you have set GOOGLE_APPLICATION_CREDENTIALS or provided the correct SERVICE_ACCOUNT_KEY_PATH.")
    exit() # Exit if initialization fails

try:
    ref = db.reference('/feedback')
    snapshot = ref.get()

    count = len(snapshot.keys()) if snapshot else 0

    print(f"Number of feedback entries: {count}")

except Exception as e:
    print(f"Error fetching feedback count: {e}")

