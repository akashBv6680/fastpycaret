# utils/email_ai_responder.py
import imaplib
import email
from email.header import decode_header
import os
import together

def fetch_latest_email():
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
        imap.select("inbox")

        status, messages = imap.search(None, '(UNSEEN)')
        if status != "OK" or not messages[0]:
            return None, None, None

        latest_email_id = messages[0].split()[-1]
        _, msg_data = imap.fetch(latest_email_id, "(RFC822)")
        raw_email = msg_data[0][1]

        msg = email.message_from_bytes(raw_email)
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")

        from_ = msg.get("From")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    charset = part.get_content_charset()
                    body = part.get_payload(decode=True).decode(charset or "utf-8")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8")

        imap.logout()
        return from_, subject, body
    except Exception as e:
        print("❌ Error fetching email:", e)
        return None, None, None

def generate_ai_reply(email_text: str) -> str:
    together.api_key = os.getenv("TOGETHER_API_KEY")

    prompt = f"""
You are an expert AI assistant. Write a polite and helpful reply to the email below.

Email:
\"\"\"
{email_text}
\"\"\"

Reply:
"""
    response = together.Complete.create(
        prompt=prompt,
        model="togethercomputer/llama-2-70b-chat",
        max_tokens=250
    )
    return response['output']['choices'][0]['text'].strip()
