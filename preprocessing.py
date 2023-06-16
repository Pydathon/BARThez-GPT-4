import pandas as pd
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('moussaKam/barthez')


class EmailCleaner:
    """
    A class to encapsulate all email cleaning functionality.
    """
    def __init__(self):
        pass

    @staticmethod
    def remove_urls(text):
        """
        Removes urls and markdown urls from the text.
        """
        url_pattern = re.compile(r'\bhttps?://\S+\b')
        markdown_url_pattern = re.compile(r'\[.*?\]\(https?://.*?\)')
        no_url = url_pattern.sub('', text)
        return markdown_url_pattern.sub('', no_url)

    @staticmethod
    def remove_spaces(email):
        """
        Remove spaces and non-printable characters from the text.
        """
        email = [i.replace("\xa0", "").replace("\u200c", "").replace("\r", "").replace("--", "").replace("  ", "").replace("**", "").replace("\xad", "") for i in email.split("\n")]
        return [i for i in email if i != '']

    def clean_email(self, email):
        """
        Apply all cleaning functions to the email.
        """
        clean = self.remove_urls(email)
        clean = self.remove_spaces(clean)
        return " ".join(clean)

    @staticmethod
    def count_tokens(email):
        """
        Counts the tokens in an email.
        """
        return len(tokenizer.encode(email))


data_to_train = pd.read_csv("path")

email_cleaner = EmailCleaner()
data_to_train["body"] = [email_cleaner.clean_email(email) for email in data_to_train["body"]]
data_to_train["count"] = [email_cleaner.count_tokens(email) for email in data_to_train["body"]]
