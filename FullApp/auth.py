import ee
import os
from pathlib import Path

def initialize_earth_engine():
    """
    Initialize Google Earth Engine with service account credentials
    """
    try:
        # Service account details
        service_account = 'your-service-account@project-id.iam.gserviceaccount.com'
        
        # Path to your service account key file
        # You can place the key.json file in a 'keys' folder in your project
        key_path = Path(__file__).parent / 'keys' / 'key.json'
        
        # Alternative: Set environment variable for the key path
        # key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', str(key_path))
        
        if not key_path.exists():
            raise FileNotFoundError(f"Service account key file not found at: {key_path}")
        
        # Initialize credentials
        credentials = ee.ServiceAccountCredentials(service_account, str(key_path))
        ee.Initialize(credentials)
        
        print("✅ Google Earth Engine initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

# Alternative: Direct initialization if you want to import and run immediately
if __name__ == "__main__":
    initialize_earth_engine()


