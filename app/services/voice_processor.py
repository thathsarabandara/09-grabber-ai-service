"""
VoiceProcessor
==============
Uses OpenAI Whisper to transcribe audio bytes and then matches the transcript
against the configured voice command phrases stored in the DB.

Transcription pipeline:
  audio bytes (WAV/MP3/etc.)
      → temp file on disk
      → Whisper model.transcribe()
      → transcript text
      → fuzzy phrase matching against DB commands
      → robot action

The Whisper model is loaded once at startup and reused across calls.
Model size is controlled by the WHISPER_MODEL setting (default: "base").
"""

import os
import tempfile
import random
from typing import List, Optional
from app.core.config import settings


class VoiceProcessor:
    def __init__(self):
        self._model = None
        self._load()

    def _load(self):
        """Load the Whisper model at service startup."""
        try:
            import whisper
            print(f"[VoiceProcessor] Loading Whisper '{settings.WHISPER_MODEL}' model…")
            self._model = whisper.load_model(settings.WHISPER_MODEL)
            print(f"[VoiceProcessor] Whisper '{settings.WHISPER_MODEL}' ready.")
        except Exception as e:
            print(f"[VoiceProcessor] Whisper load failed: {e}")
            self._model = None

    # ── Transcription ──────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """
        Transcribe raw audio bytes (WAV, MP3, WebM, etc.) using Whisper.
        Returns the transcript string, or None on failure.
        """
        if self._model is None:
            return None

        # Write to a temp file so Whisper can read it via ffmpeg
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self._model.transcribe(tmp_path, fp16=False)
            return result.get("text", "").strip().lower()
        except Exception as e:
            print(f"[VoiceProcessor] Transcription error: {e}")
            return None
        finally:
            os.unlink(tmp_path)

    # ── Intent matching ────────────────────────────────────────────────────

    def _match_command(self, transcript: str, commands: List) -> Optional[object]:
        """
        Find the best-matching command for the transcript using simple
        substring scoring. Returns the command object or None.
        """
        if not transcript:
            return None

        best_cmd = None
        best_score = 0

        for cmd in commands:
            phrase_lower = cmd.phrase.lower()
            # Count how many words in the phrase appear in the transcript
            words = phrase_lower.split()
            matches = sum(1 for w in words if w in transcript)
            score = matches / max(len(words), 1)
            if score > best_score:
                best_score = score
                best_cmd = cmd

        # Require at least 50% word overlap to count as a match
        return best_cmd if best_score >= 0.5 else None

    def match_intent(self, audio_bytes: bytes, commands: List) -> Optional[dict]:
        """
        Full pipeline: transcribe audio → match against commands → return result.
        Used by the POST /voice/transcribe endpoint.
        """
        transcript = self.transcribe(audio_bytes)
        if transcript is None:
            return None

        matched = self._match_command(transcript, commands)
        if not matched:
            return {
                "transcript": transcript,
                "phrase": None,
                "intent": "NO_MATCH",
                "action": "No matching command found"
            }

        intent_map = {
            "RELEASE": "RELEASE_OBJECT",
            "GRAB": "GRAB_OBJECT",
            "MOVE_LEFT": "MOVE_LEFT_COMMAND",
            "MOVE_RIGHT": "MOVE_RIGHT_COMMAND",
            "MOVE_FORWARD": "MOVE_FORWARD_COMMAND",
            "MOVE_BACKWARD": "MOVE_BACKWARD_COMMAND",
            "MOVE_UP": "MOVE_UP_COMMAND",
            "MOVE_DOWN": "MOVE_DOWN_COMMAND",
            "MOVE_HOME": "MOVE_HOME_COMMAND",
            "STOP_ALL": "EMERGENCY_STOP_COMMAND",
            "Pick And Place": "PICK_AND_PLACE_OBJECT",
            "Smart Sorting": "SMART_SORTING_COMMAND",
        }
        intent = intent_map.get(matched.action, f"EXECUTE_{matched.action.upper()}")

        return {
            "transcript": transcript,
            "phrase": matched.phrase,
            "intent": intent,
            "action": f"Executing: {matched.action}({matched.target or 'any'})"
        }

    def parse_intent(self, commands: List) -> Optional[dict]:
        """
        Simulation fallback used by the /voice/test endpoint (no audio input).
        Picks a random command and formats the result the same way as match_intent.
        """
        if not commands:
            return None

        chosen = random.choice(commands)
        intent_map = {
            "RELEASE": "RELEASE_OBJECT",
            "GRAB": "GRAB_OBJECT",
            "MOVE_LEFT": "MOVE_LEFT_COMMAND",
            "MOVE_RIGHT": "MOVE_RIGHT_COMMAND",
            "MOVE_FORWARD": "MOVE_FORWARD_COMMAND",
            "MOVE_BACKWARD": "MOVE_BACKWARD_COMMAND",
            "MOVE_UP": "MOVE_UP_COMMAND",
            "MOVE_DOWN": "MOVE_DOWN_COMMAND",
            "MOVE_HOME": "MOVE_HOME_COMMAND",
            "STOP_ALL": "EMERGENCY_STOP_COMMAND",
            "Pick And Place": "PICK_AND_PLACE_OBJECT",
            "Smart Sorting": "SMART_SORTING_COMMAND",
        }
        intent = intent_map.get(chosen.action, f"EXECUTE_{chosen.action.upper()}")
        return {
            "transcript": f"[simulated] {chosen.phrase}",
            "phrase": chosen.phrase,
            "intent": intent,
            "action": f"Executing: {chosen.action}({chosen.target or 'any'})"
        }
