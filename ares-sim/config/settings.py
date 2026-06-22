from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str
    model_1: str
        
    model_config = SettingsConfigDict(env_file=".env")


ZONE_CONTEST_RATIO= 0.9
ZONE_3_WIN_THRESHOLD=6

ADJACENCY = {
    1: [2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4],
}

settings = Settings()

