"""Voice I/O for the radio wingman, with graceful degradation.

Speech-to-text uses the ``SpeechRecognition`` package when installed
(``pip install SpeechRecognition``; add ``openai-whisper`` for offline
recognition, otherwise the free Google web recognizer is used).
Text-to-speech uses ``pyttsx3`` (offline, uses Windows SAPI voices on the
DCS machine).  When either is missing, the radio falls back to the console:
type transmissions at the ``[RADIO] >`` prompt, replies are printed.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional


class VoiceIO:
    def __init__(self, prefer_voice: bool = True, language: str = "ko-KR"):
        self.language = language
        self._recognizer = None
        self._microphone = None
        self._tts = None
        # Speech runs on a dedicated worker so say() never blocks its caller
        # (the 20 Hz flight loop makes proactive calls; a blocking runAndWait()
        # there would stall control output and trip the Lua command timeout).
        self._speech_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._speech_thread: Optional[threading.Thread] = None

        if prefer_voice:
            try:
                import speech_recognition as sr

                self._recognizer = sr.Recognizer()
                self._microphone = sr.Microphone()
                with self._microphone as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("radio: microphone ready (speak after the prompt)")
            except Exception as exc:
                print(f"radio: no microphone/STT ({exc}); type transmissions instead")

            try:
                import pyttsx3

                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 175)
                self._speech_thread = threading.Thread(
                    target=self._speech_worker, daemon=True, name="radio-tts"
                )
                self._speech_thread.start()
            except Exception as exc:
                print(f"radio: no TTS ({exc}); replies will be printed only")

    # ------------------------------------------------------------------ #
    def listen(self) -> Optional[str]:
        """Blocking: one transmission from the flight lead (voice or console)."""
        if self._recognizer is not None and self._microphone is not None:
            import speech_recognition as sr

            print("[RADIO] listening...")
            try:
                with self._microphone as source:
                    audio = self._recognizer.listen(source, phrase_time_limit=6)
                try:
                    return self._recognizer.recognize_whisper(audio, language=self.language[:2])
                except (AttributeError, ImportError):
                    return self._recognizer.recognize_google(audio, language=self.language)
            except sr.UnknownValueError:
                return None
            except Exception as exc:
                print(f"radio: STT error ({exc}); falling back to console input")
                self._recognizer = None

        try:
            text = input("[RADIO] > ").strip()
        except EOFError:
            return None
        return text or None

    def say(self, text: str) -> None:
        """Print a radio reply immediately and queue it to be spoken.

        Non-blocking and thread-safe: the caller (including the 20 Hz flight
        loop) returns at once while a dedicated worker speaks queued lines
        serially, so no two calls overlap and control output never stalls.
        """
        print(f"[RADIO] {text}")
        if self._speech_thread is not None:
            self._speech_queue.put(text)

    def _speech_worker(self) -> None:
        while True:
            text = self._speech_queue.get()
            if text is None:  # shutdown sentinel
                return
            if self._tts is None:
                continue
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as exc:
                print(f"radio: TTS error ({exc}); continuing text-only")
                self._tts = None

    def close(self) -> None:
        """Stop the speech worker (best effort)."""
        if self._speech_thread is not None:
            self._speech_queue.put(None)
