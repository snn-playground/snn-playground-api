from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Kaptios SNN Visualizer API"
    app_description: str = "API for the Kaptios SNN visualization application"
    version: str = "0.0.0"

    class Config:
        env_file = '.env.example', '.env'

settings = Settings()