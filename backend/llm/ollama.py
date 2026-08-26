import json
import os

from ollama import AsyncClient

from backend.services.models import ActionType, EmailDigest, EmailMessage


# Override without touching code:  LLM_MODEL=digest-cfgfix
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

# qwen3 reasons before answering, which costs far more tokens than the
# digest itself at ~6 tok/s locally, so it is turned off. Models with
# no thinking mode (qwen2.5 and the fine-tune built on it) reject the
# flag outright, and the client omits the field entirely when None.
DEFAULT_THINK = False if "qwen3" in DEFAULT_MODEL else None


class OllamaService:
    def __init__(
        self,
        model: str | None = None,
        host: str = "http://localhost:11434",
        think: bool | None = ...,
    ):
        self.model = model or DEFAULT_MODEL

        self.think = (
            DEFAULT_THINK if think is ... else think
        )
        # No timeout on purpose. Local generation is slow and a
        # cap just kills long-but-healthy calls (a 300s one made
        # every digest fail). None is also ollama's own default.
        self.client = AsyncClient(host=host, timeout=None)

    async def summarize_emails(
        self,
        emails: list[EmailMessage],
    ) -> EmailDigest:

        email_data = [
            {
                "sender": email.sender_name,
                "sender_email": email.sender_email,
                "subject": email.subject,
                "snippet": email.snippet,
                "timestamp": email.timestamp,
            }
            for email in emails
        ]

        prompt = f"""
        You are an email triage assistant.

        From the emails below, pick out only the ones where the user
        has to DO something.

        Include: a bill or payment with an amount or due date, an OTP
        or code to hand over, a security alert about account access, a
        message someone is waiting on a reply to, anything with a
        stated deadline.

        Exclude completely: newsletters, promotions and offers, job
        alerts and recommendations, social notifications, and receipts
        for things already done. Leave these out entirely -- do not
        include them as low priority.

        For each email that qualifies, give:
        - action: the concrete thing the user must do
        - reason: why it matters, citing specifics from the email
          (amounts, dates, deadlines, device names)
        - priority: high, medium or low
        - action_type: "required" if ignoring it has a real
          consequence, otherwise "recommended"

        Then give:
        - summary: two or three sentences on what actually needs
          attention. Do not list who the emails are from.
        - action_items: the action text of the "required" items only.

        Most inboxes yield only a few real items. If nothing needs
        action, return an empty priority_items list. Use only what is
        stated in the emails.

        Emails:

        {json.dumps(email_data, indent=2)}
        """

        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=EmailDigest.model_json_schema(),
            think=self.think,
        )

        content = response["message"]["content"]

        try:
            data = json.loads(content)
            digest = EmailDigest.model_validate(data)

            # Derive rather than trust: the model tends to either
            # mirror priority_items verbatim here or disagree with
            # its own action_type labels.
            digest.action_items = [
                item.action
                for item in digest.priority_items
                if item.action_type is ActionType.REQUIRED
            ]

            return digest

        except Exception as exc:
            print("\n===== INVALID LLM OUTPUT =====")
            print(content)
            print("==============================")
            raise ValueError(
                f"LLM returned invalid EmailDigest: {exc}"
            ) from exc