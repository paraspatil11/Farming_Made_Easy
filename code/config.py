from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    # Appwrite settings
    APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
    APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
    APPWRITE_BUCKET_ID = os.getenv("APPWRITE_BUCKET_ID")