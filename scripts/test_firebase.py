"""Test Firebase connection and save a test record."""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

import firebase_admin
from firebase_admin import credentials, firestore

def test_firebase():
    print("=" * 50)
    print("Firebase Connection Test")
    print("=" * 50)
    
    # Check credentials
    cred_file = os.environ.get('FIREBASE_CREDENTIALS_FILE')
    print(f"\nCredentials file: {cred_file}")
    
    if cred_file:
        if os.path.exists(cred_file):
            print(f"  File exists: YES")
        else:
            print(f"  File exists: NO - This is the problem!")
            return False
    else:
        print("  No FIREBASE_CREDENTIALS_FILE set")
        
        # Check other options
        if os.environ.get('FIREBASE_SERVICE_ACCOUNT_BASE64'):
            print("  FIREBASE_SERVICE_ACCOUNT_BASE64 is set")
        elif os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON'):
            print("  FIREBASE_SERVICE_ACCOUNT_JSON is set")
        else:
            print("  No Firebase credentials found!")
            return False
    
    # Try to initialize Firebase
    print("\nInitializing Firebase...")
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_file)
            firebase_admin.initialize_app(credential=cred)
        print("  Firebase initialized: SUCCESS")
    except Exception as e:
        print(f"  Firebase initialization FAILED: {e}")
        return False
    
    # Try to get Firestore client
    print("\nGetting Firestore client...")
    try:
        db = firestore.client()
        print("  Firestore client: SUCCESS")
    except Exception as e:
        print(f"  Firestore client FAILED: {e}")
        return False
    
    # Try to write a test document
    print("\nWriting test document...")
    try:
        test_data = {
            'test': True,
            'timestamp': datetime.now(timezone.utc),
            'message': 'Connection test from headless detector',
        }
        
        doc_ref = db.collection('test_connection').document()
        doc_ref.set(test_data)
        print(f"  Document written: SUCCESS (ID: {doc_ref.id})")
        
        # Clean up test document
        doc_ref.delete()
        print(f"  Test document deleted")
        
    except Exception as e:
        print(f"  Write FAILED: {e}")
        return False
    
    # Try to write to the actual collection
    print("\nWriting to utensil_sessions collection...")
    try:
        session_data = {
            'utensil_type': 'rounded_rectangle',
            'average_volume_ml': 150.0,
            'average_percent_fill': 75.0,
            'sample_count': 5,
            'detected_food': 'rice',
            'food_confidence': 0.95,
            'started_at': datetime.now(timezone.utc),
            'ended_at': datetime.now(timezone.utc),
            'created_at': datetime.now(timezone.utc),
            'is_test': True,  # Mark as test
        }
        
        doc_ref = db.collection('utensil_sessions').document()
        doc_ref.set(session_data)
        print(f"  Session saved: SUCCESS (ID: {doc_ref.id})")
        print(f"\n  NOTE: Test session was saved. You can delete it manually if needed.")
        
    except Exception as e:
        print(f"  Session save FAILED: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All tests PASSED - Firebase is working!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_firebase()
    sys.exit(0 if success else 1)
