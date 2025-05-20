from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import firebase_admin
from firebase_admin import db

class FeedbackHandler:
    def __init__(self, database_url: str = None):
        if not firebase_admin._apps:
            cred = firebase_admin.credentials.Certificate('firebase-credentials.json')
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })
        self.db = db.reference()
        self.feedback_ref = self.db.child('feedback')

    def store_feedback(self, feedback_data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = datetime.now().isoformat()
            
            new_feedback_ref = self.feedback_ref.push(feedback_data)
            return True, "✅ Feedback submitted successfully!"
            
        except Exception as e:
            print(f"Firebase Error: {str(e)}")
            return False, f"❌ Error: {str(e)}"

    def get_preferred_responses(self) -> Dict[str, int]:
        """Get count of preferred responses (A vs B)."""
        counts = {'A': 0, 'B': 0}
        try:
            feedback_data = self.feedback_ref.get()
            if feedback_data:
                for entry in feedback_data.values():
                    preferred = entry.get('responses', {}).get('selected', '').replace('Response ', '')
                    if preferred in counts:
                        counts[preferred] += 1
        except Exception as e:
            print(f"Error getting preferred responses: {str(e)}")
        return counts

    def get_average_ratings(self) -> Dict[str, float]:
        total_quality = 0
        total_speed = 0
        count = 0
        
        try:
            feedback_data = self.feedback_ref.get()
            if feedback_data:
                for entry in feedback_data.values():
                    ratings = entry.get('ratings', {})
                    total_quality += float(ratings.get('overall_quality', 0))
                    total_speed += float(ratings.get('response_speed', 0))
                    count += 1
        except Exception as e:
            print(f"Error calculating averages: {str(e)}")
            
        if count == 0:
            return {'quality': 0.0, 'speed': 0.0}
            
        return {
            'quality': total_quality / count,
            'speed': total_speed / count
        }