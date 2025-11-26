import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment
postmark_token = os.getenv("POSTMARK_SERVER_TOKEN")
from_email = os.getenv("EMAIL_FROM", "noreply@yourdomain.com")
test_email = os.getenv("EMAIL_TEST_DESTINATION", "test@example.com")

# Data for the API request
url = "https://api.postmarkapp.com/email"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "X-Postmark-Server-Token": postmark_token
}
data = {
    "From": from_email,
    "To": test_email,
    "Subject": "Hello from Postmark",
    "HtmlBody": "<strong>Hello</strong> dear Postmark user.",
    "MessageStream": "outbound"
}

# Send the email
response = requests.post(url, json=data, headers=headers)

# Verify the response
if response.status_code == 200:
    print("Email sent successfully")
else:
    print(f"Error sending email: {response.status_code} - {response.text}")
