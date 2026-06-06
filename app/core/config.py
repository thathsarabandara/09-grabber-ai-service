from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Grabber Service"
    DATABASE_URL: str = "mysql+pymysql://user:password@localhost/dbname"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
