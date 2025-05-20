import os
import firebase_admin
from firebase_admin import credentials, db

class FirebaseManager:
    def __init__(self, credentials_path=None):
        if not credentials_path:
            raise FileNotFoundError("Firebase credentials path not provided!")
            
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Firebase credentials not found at: {credentials_path}")
            
        cred = credentials.Certificate(credentials_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://rlhf-2e2cc-default-rtdb.europe-west1.firebasedatabase.app'
            })
        
        self.db = db.reference()
        self.feedback_ref = self.db.child('feedback')

    def store_feedback(self, feedback_data: dict) -> bool:
        """Store feedback in Realtime Database"""
        try:
            self.feedback_ref.push(feedback_data)
            return True
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return False

    def fetch_feedback(self):
        """Retrieve feedback from Realtime Database"""
        try:
            feedback_data = self.feedback_ref.get()
            if feedback_data:
                return feedback_data
            return {}
        except Exception as e:
            print(f"Error fetching feedback: {e}")
            return {}