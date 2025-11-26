import requests
from requests.auth import HTTPBasicAuth

def get_paypal_access_token(client_id, client_secret):
    url = "https://api.sandbox.paypal.com/v1/oauth2/token"
    headers = {
        "Accept": "application/json",
        "Accept-Language": "en_US",
    }
    data = {
        "grant_type": "client_credentials"
    }
    response = requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(client_id, client_secret))
    return response.json().get("access_token")


def verify_paypal_transaction(access_token, order_id):
    url = f"https://api.sandbox.paypal.com/v2/checkout/orders/{order_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    return response.json()
