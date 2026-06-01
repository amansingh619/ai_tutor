import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings:
    def __init__(self):
        self.LIVEKIT_URL = os.getenv("LIVEKIT_URL")
        self.LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
        self.LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
        self.WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
        self.WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        self.PIPER_EXECUTABLE = os.getenv("PIPER_EXECUTABLE", "piper")
        self.PIPER_MODELS_DIR = os.getenv(
            "PIPER_MODELS_DIR", os.path.join(BASE_DIR, "models")
        )
        self.PIPER_DEFAULT_LANGUAGE = os.getenv("PIPER_DEFAULT_LANGUAGE", "en")

        self.TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "Spanish")
        self.NATIVE_LANGUAGE = os.getenv("NATIVE_LANGUAGE", "English")

        self.DB_PATH = os.getenv("DB_PATH", "tutor.db")

    def validate(self):
        """Validating whether a required env variiables exist or not
        """
        missing = []
        if not self.HF_TOKEN:
            missing.append("HF_TOKEN")
        if not self.LIVEKIT_API_KEY:
            missing.append("LIVEKIT_API_KEY")
        if not self.LIVEKIT_API_SECRET:
            missing.append("LIVEKIT_API_SECRET")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


settings = Settings()
