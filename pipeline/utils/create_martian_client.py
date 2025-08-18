import os
from martian_apart_hack_sdk import martian_client
from dotenv import load_dotenv

load_dotenv()

def create_martian_client():
    return martian_client.MartianClient(api_url=os.getenv('MARTIAN_API_URL'), api_key=os.getenv('MARTIAN_API_KEY'))
