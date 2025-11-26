import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config(object):
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    TESTING = os.getenv("TESTING", "False").lower() == "true"

class DevelopmentConfig(Config):
    # Security
    SECRET_KEY = os.getenv("APP_SECRET_KEY", "default-secret-key-change-me")

    # API Keys
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    ELEVEN_KEY = os.getenv("ELEVEN_KEY")
    TWILIO_SID = os.getenv("TWILIO_SID")
    TWILIO_AUTH = os.getenv("TWILIO_AUTH")
    DEEPGRAM_KEY = os.getenv("DEEPGRAM_KEY")

    # Models
    MODEL_GPT = os.getenv("MODEL_GPT", "gpt-4-turbo")

    # Service Configuration
    TTS_GENERATOR = os.getenv("TTS_GENERATOR", "elevenlabs")
    STT_GENERATOR = os.getenv("STT_GENERATOR", "whisper")

config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}
