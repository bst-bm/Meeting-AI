"""Meeting minutes extraction via Ollama LLM."""

import json
import re
import requests


PROMPT = """You are an expert meeting analyst. Analyze the transcript below and extract structured meeting minutes.

TRANSCRIPT:
{transcript}

Return ONLY a valid JSON object with this exact structure (no explanation, no markdown):
{{
  "title": "Short descriptive meeting title",
  "summary": "2-3 sentence executive summary",
  "participants": ["Full Name or Speaker label", "..."],
  "topics": [
    {{"title": "Topic name", "summary": "Brief description of what was discussed"}}
  ],
  "action_points": [
    {{"action": "Concrete task to be done", "owner": "Person responsible (or 'TBD')", "deadline": "Deadline if mentioned (or null)"}}
  ],
  "decisions": ["Decision 1", "Decision 2"],
  "next_meeting": "Next meeting info if mentioned, otherwise null"
}}"""


class MinutesGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "mistral:latest"):
        self.api_url = f"{ollama_url}/api/generate"
        self.model = model

    def generate(self, transcript: str) -> dict:
        print(f"  Sending transcript to Ollama ({self.model})...")
        prompt = PROMPT.format(transcript=transcript)

        response = requests.post(
            self.api_url,
            json={
                "model":       self.model,
                "prompt":      prompt,
                "stream":      False,
                "temperature": 0.1,
                "format":      "json",
                "options": {
                    "num_ctx": 32768,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"LLM did not return valid JSON:\n{raw[:400]}")
