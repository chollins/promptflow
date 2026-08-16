from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from requests.auth import HTTPBasicAuth


@dataclass(frozen=True)
class EmailResult:
    sent: bool
    provider: str | None = None
    status_code: int | None = None
    response_text: str | None = None


def _mailjet_settings() -> dict[str, str]:
    return {
        "api_key": os.getenv("MAILJET_API_KEY", "").strip(),
        "secret_key": os.getenv("MAILJET_SECRET_KEY", "").strip(),
        "from_email": os.getenv("MAILJET_FROM_EMAIL", "").strip(),
        "from_name": os.getenv("MAILJET_FROM_NAME", "PromptFlow Support").strip(),
    }


def _mailjet_ready() -> bool:
    settings = _mailjet_settings()
    return all([settings["api_key"], settings["secret_key"], settings["from_email"]])


def send_email(*, to_email: str, to_name: str | None, subject: str, text_body: str) -> EmailResult:
    settings = _mailjet_settings()
    if not _mailjet_ready():
        return EmailResult(sent=False, provider="mailjet")

    payload = {
        "Messages": [
            {
                "From": {
                    "Email": settings["from_email"],
                    "Name": settings["from_name"],
                },
                "To": [
                    {
                        "Email": to_email,
                        "Name": to_name or to_email,
                    }
                ],
                "Subject": subject,
                "TextPart": text_body,
            }
        ]
    }

    response = requests.post(
        "https://api.mailjet.com/v3.1/send",
        json=payload,
        auth=HTTPBasicAuth(settings["api_key"], settings["secret_key"]),
        timeout=10,
    )
    response.raise_for_status()
    return EmailResult(
        sent=True,
        provider="mailjet",
        status_code=response.status_code,
        response_text=response.text,
    )


def send_password_reset_email(*, to_email: str, to_name: str, otp_code: str) -> EmailResult:
    return send_email(
        to_email=to_email,
        to_name=to_name,
        subject="PromptFlow Password Reset Code",
        text_body=(
            f"Your password reset code is:\n\n{otp_code}\n\n"
            "This code expires in 10 minutes.\n\n"
            "If you did not request this, you can safely ignore this email."
        ),
    )


def send_invitation_email(*, to_email: str, to_name: str, registration_link: str) -> EmailResult:
    return send_email(
        to_email=to_email,
        to_name=to_name,
        subject="You're invited to PromptFlow",
        text_body=(
            f"You have been invited to join PromptFlow.\n\n"
            f"Register here:\n{registration_link}\n\n"
            "This invitation expires in 7 days."
        ),
    )
