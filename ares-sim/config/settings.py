from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    model_1: str
    



ZONE_CONTEST_RATIO= 0.9
ZONE_3_WIN_THRESHOLD=6

ADJACENCY = {
    1: [2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4],
}

