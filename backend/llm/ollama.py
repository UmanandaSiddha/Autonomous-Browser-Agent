import json

from ollama import AsyncClient

from backend.services.models import EmailDigest, EmailMessage


class OllamaService:
    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        # Local models are slow; this only guards a real hang.
        self.client = AsyncClient(host=host, timeout=300)

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
        You are an email intelligence assistant.

        Analyze the following recent emails.

        Your job is to:

        1. Provide a concise overall summary.
        2. Identify emails that genuinely require attention.
        3. Assign each important item a priority:
        - high
        - medium
        - low
        4. Explain why each item matters.
        5. Identify concrete actions the user actually needs to perform.
        6. Do not invent actions that are not supported by the email.
        7. Do not turn purely informational emails into required actions.
        8. Ignore promotional emails unless they contain a meaningful action.

        Important:
        - "action" must describe a concrete action.
        - "reason" must explain why the email requires attention.
        - "action_type" must be either "required" or "recommended".
        - Do not create an action item when no action is necessary.

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
        )

        content = response["message"]["content"]

        try:
            data = json.loads(content)
            return EmailDigest.model_validate(data)

        except Exception as exc:
            print("\n===== INVALID LLM OUTPUT =====")
            print(content)
            print("==============================")
            raise ValueError(
                f"LLM returned invalid EmailDigest: {exc}"
            ) from exc