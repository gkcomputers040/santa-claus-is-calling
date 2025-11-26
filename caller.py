import os
import sys
import asyncio
import sqlite3
from twilio.rest import Client
from dotenv import load_dotenv
#import config

load_dotenv()

# Twilio configuration
account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH')
twilio_number = os.getenv('TWILIO_NUMBER')
dbname = os.getenv('DATABASE')
call_base_url = os.getenv('CALL_BASE_URL')
twilio_client = Client(account_sid, auth_token)

# Function to initiate call
async def initiate_call(user_id, phone_number, call_job_id):
    call = twilio_client.calls.create(
        to=phone_number,
        from_=twilio_number,
        url=f'{call_base_url}/answer/{user_id}/{call_job_id}'
    )

# Function to get user data from the SQLite database and validate
def get_user_data(user_id):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute("""
                SELECT user_details.phone_number, calls.call_job_id
                FROM users
                INNER JOIN user_details ON users.id = user_details.user_id
                INNER JOIN calls ON users.id = calls.user_id
                WHERE users.id = ?
              """, (user_id,))
    data = c.fetchone()
    conn.close()
    return data

# Main function
if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
        user_data = get_user_data(user_id)
        if user_data:
            phone_number, call_job_id = user_data
            print("Calling..")
            asyncio.run(initiate_call(user_id, phone_number, call_job_id))
        else:
            print("Invalid user_id or user does not exist.")
    else:
        print("No user_id provided.")
