from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import os.path
import pickle
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import pandas as pd
import time
from bs4 import BeautifulSoup
from langdetect import detect
import re


class GmailClient:
    def __init__(self, user_email, token_file='token.pickle', cred_file='desktop_creds.json'):
        self.user_email = user_email
        self.token_file = token_file
        self.cred_file = cred_file
        self.creds = self.get_credentials()
        self.service = build('gmail', 'v1', credentials=self.creds)
        
    def get_credentials(self):
        """
        Get the user credentials for the Gmail API.

        :param token_file: path to the token file.
        :param cred_file: path to the credentials file.
        :return: creds object.
        """
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'desktop_creds.json', ['https://www.googleapis.com/auth/gmail.readonly'])
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return creds

    def process_parts(self, parts):
        """
        Process the parts of an email message.

        :param parts: list of parts of an email message.
        :return: list of decoded text from the parts.
        """
        text_list = []

        for part in parts:
            mimeType = part.get('mimeType')
            body = part.get('body')
            data = body.get('data')

            if mimeType == "text/plain":
                text = base64.urlsafe_b64decode(data).decode('utf-8')
                text_list.append(text)
            elif mimeType == "text/html":
                soup = BeautifulSoup(base64.urlsafe_b64decode(data), "lxml")
                text = soup.get_text()
                text_list.append(text)
            elif mimeType in ["multipart/alternative", "multipart/mixed", 'multipart/related']:
                if part.get('parts'):
                    nested_text_list = self.process_parts(part['parts'])
                    text_list.extend(nested_text_list)

        return text_list

    def get_email_body(self, service, msg):
        """
        Get the body of an email message.

        :param service: the Gmail API service object.
        :param msg: the email message object.
        :return: the body of the email message as a string.
        """

        text_list = []
        mimeType = msg['payload']['mimeType']

        if mimeType == 'text/plain':
            email_body = msg['payload']['body']['data']
            text = base64.urlsafe_b64decode(email_body).decode('utf-8')
            text_list.append(text)
        elif mimeType == 'text/html':
            email_body = msg['payload']['body']['data']
            soup = BeautifulSoup(base64.urlsafe_b64decode(email_body), "lxml")
            text = soup.get_text()
            text_list.append(text)
        elif mimeType in ['multipart/alternative', 'multipart/mixed', 'multipart/related']:
            text_list.extend(self.process_parts(msg['payload']['parts']))

        return ' '.join(text_list)


    def get_email_data(self, service, message):
        """
        Get the data of an email message.

        :param service: the Gmail API service object.
        :param message: the email message object.
        :return: dictionary with the data
        of the email message.
        """

        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        email_body = self.get_email_body(service, msg)
        headers = msg['payload']['headers']
        payload = msg['payload']

        subject = None
        sender_name = None
        sender_adress = None
        date = None

        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                from_data = self.split_name_email(header['value'])
                sender_name = from_data[0].replace('"', "")
                sender_adress = from_data[1]
            elif header['name'] == 'Date':
                date = header['value']

        return {
            'raw':payload,
            'subject': subject,
            'sender_name': sender_name,
            'sender_adress': sender_adress,
            'date': date,
            'body': email_body
        }

    def parse_emails_to_dataframe(self, messages, service):
        """
        Parse a list of email messages to a DataFrame.

        :param messages: list of email messages.
        :param service: the Gmail API service object.
        :return: DataFrame with the email data.
        """
        email_data = []
        for message in messages:
            email_data.append(self.get_email_data(service, message))
        df = pd.DataFrame(email_data)
        return df
    
    def split_name_email(self, input_string):
        """
        Split email name of the sender and email adress of the sender
        
        """
        match = re.match(r'(.*)\s*<(.*)>', input_string)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
        else:
            return input_string


    def get_emails_last_24h(self):
        try:
            creds = self.get_credentials()
            service = build('gmail', 'v1', credentials=creds)

            now = time.time()
            yesterday = now - 60 * 60 * 24
            query = f'after:{int(yesterday)}'

            results = service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])

            if not messages:
                print('No emails found in the last 24 hours.')
            else:
                print(f'Emails from the last 24 hours:')
                return self.parse_emails_to_dataframe(messages, service)

        except HttpError as error:
            print(f'An error occurred: {error}')
            
    def get_emails_last_n(self, n):
        """
        Get the last n emails not sent by the user.

        :param n: number of emails to get.
        :return: DataFrame with the email data.
        """

        creds = self.get_credentials()
        service = build('gmail', 'v1', credentials=creds)

        response = service.users().messages().list(userId='me').execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        while 'nextPageToken' in response and len(messages) < n:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId='me', pageToken=page_token).execute()
            messages.extend(response['messages'])
        
        fail_data = []
        email_data = []
        for message in messages:
            try:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                email_info = self.get_email_data(service, msg)
                # skip the email if it's sent by the user
                if email_info['sender_adress'] == self.user_email:
                    continue
                email_data.append(email_info)
                if len(email_data) == n:
                    break
            except:
                fail_data.append(message)
                continue

        df = pd.DataFrame(email_data)
        lang = []
        for i in df["body"]:
            try:
                lang.append(detect(i))
            except:
                lang.append("")
        df["lang"] = lang
        return df, fail_data





# Usage:
client = GmailClient(user_email="email")
data, fails = client.get_emails_last_n(1000)
