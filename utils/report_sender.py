# utils/report_sender.py
import smtplib
import os
import mimetypes
from email.message import EmailMessage
import re

def is_valid_email(email: str) -> bool:
    """Validates email format."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def send_email_with_attachments(to_email, subject, body, file_paths=None) -> bool:
    """Send email with attachments via Gmail."""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = os.getenv("EMAIL_ADDRESS")
        msg["To"] = to_email
        msg.set_content(body)

        file_paths = file_paths or []

        for file_path in file_paths:
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type, mime_subtype = mime_type.split('/')
            with open(file_path, 'rb') as file:
                msg.add_attachment(file.read(),
                                   maintype=mime_type,
                                   subtype=mime_subtype,
                                   filename=os.path.basename(file_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
            smtp.send_message(msg)

        return True

    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False
