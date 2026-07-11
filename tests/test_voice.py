import time
import unittest

from dcs_bridge.voice import VoiceIO


class _SlowTTS:
    """Fake pyttsx3 engine whose runAndWait blocks, to prove say() doesn't."""

    def __init__(self, delay=0.2):
        self.delay = delay
        self.spoken = []

    def setProperty(self, *_):
        pass

    def say(self, text):
        self._pending = text

    def runAndWait(self):
        time.sleep(self.delay)
        self.spoken.append(self._pending)


class VoiceIOTest(unittest.TestCase):
    def _voice_with_tts(self, tts):
        import threading

        v = VoiceIO(prefer_voice=False)  # no mic, no real TTS
        v._tts = tts
        v._speech_thread = threading.Thread(
            target=v._speech_worker, daemon=True, name="test-tts"
        )
        v._speech_thread.start()
        return v

    def test_say_does_not_block_caller(self):
        tts = _SlowTTS(delay=0.3)
        v = self._voice_with_tts(tts)

        start = time.monotonic()
        for phrase in ("2, tally.", "2, engaging.", "2, guns."):
            v.say(phrase)
        elapsed = time.monotonic() - start
        # Three 0.3 s phrases would be 0.9 s if say() blocked; it must not.
        self.assertLess(elapsed, 0.1, f"say() blocked the caller ({elapsed:.2f}s)")

        v.close()
        v._speech_thread.join(timeout=3.0)
        self.assertEqual(tts.spoken, ["2, tally.", "2, engaging.", "2, guns."])

    def test_console_only_say_is_safe(self):
        v = VoiceIO(prefer_voice=False)
        v.say("no tts configured")  # must not raise, no worker thread
        self.assertIsNone(v._speech_thread)


if __name__ == "__main__":
    unittest.main()
