"""Meeting minutes extraction via Ollama LLM."""

import json
import re
import requests

_NUM_CTX = 32768
_RESPONSE_RESERVE = 1500  # tokens kept free for JSON output
_CHARS_PER_TOKEN = 4      # heuristic for western languages (~4 chars/token)

_PROMPT = """You are an expert meeting analyst. Analyze the transcript below and extract structured meeting minutes.
{speakers_instruction}
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

_SPEAKERS_INSTRUCTION = """
IMPORTANT: The following {n} speakers were detected in the transcript.
List ALL of them in the participants array — do not omit any, even if they spoke only briefly:
{speaker_list}
"""


class MinutesGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:32b-instruct-q4_K_M"):
        self.api_url = f"{ollama_url}/api/generate"
        self._show_url = f"{ollama_url}/api/show"
        self.model = model
        self.num_ctx = self._resolve_context_size()

    def _resolve_context_size(self) -> int:
        """Query Ollama for the model's native context length, cap at _NUM_CTX."""
        try:
            r = requests.post(self._show_url, json={"name": self.model}, timeout=10)
            r.raise_for_status()
            model_info = r.json().get("model_info", {})
            for key, value in model_info.items():
                if key.endswith("context_length"):
                    return min(_NUM_CTX, int(value))
        except Exception:
            pass
        return _NUM_CTX

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // _CHARS_PER_TOKEN

    def _check_context(self, prompt: str) -> None:
        estimated = self._estimate_tokens(prompt)
        available = self.num_ctx - _RESPONSE_RESERVE
        usage_pct = estimated / available

        if estimated > available:
            overage = estimated - available
            print(
                f"\n  WARNING: Transcript exceeds model context window!\n"
                f"  Estimated tokens : ~{estimated:,}\n"
                f"  Available tokens : ~{available:,}  "
                f"(context={self.num_ctx:,}, reserved={_RESPONSE_RESERVE})\n"
                f"  Overflow         : ~{overage:,} tokens — the end of the transcript will be cut off.\n"
                f"  Consider using --model with a larger-context model or --no-diarize to shorten the transcript.\n"
            )
        elif usage_pct > 0.85:
            print(
                f"  Note: Transcript is large (~{estimated:,} tokens, {usage_pct:.0%} of available context)."
            )

    def _extract_speakers(self, transcript: str) -> list[str]:
        return sorted(set(re.findall(r"^(SPEAKER_\d+):", transcript, re.MULTILINE)))

    def generate(self, transcript: str) -> dict:
        speakers = self._extract_speakers(transcript)
        if speakers:
            speakers_instruction = _SPEAKERS_INSTRUCTION.format(
                n=len(speakers),
                speaker_list=", ".join(speakers),
            )
        else:
            speakers_instruction = ""

        prompt = _PROMPT.format(transcript=transcript, speakers_instruction=speakers_instruction)
        self._check_context(prompt)
        print(f"  Sending transcript to Ollama ({self.model})...")

        response = requests.post(
            self.api_url,
            json={
                "model":       self.model,
                "prompt":      prompt,
                "stream":      False,
                "temperature": 0.1,
                "format":      "json",
                "options": {
                    "num_ctx": self.num_ctx,
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
