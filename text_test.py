import os
from twilio.rest import Client


accountSid = os.environ.get("TWILIO_ACCOUNT_SID")
authToken = os.environ.get("TWILIO_AUTH_TOKEN")
client = Client(accountSid, authToken)

client.messages.create(
    to=os.environ.get("MY_PHONE_NUMBER"),
    from_="7204109188",
    body="Script is finished!"
)
