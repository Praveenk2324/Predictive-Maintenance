from src.api.database import engine, Base
from src.api import models

print("Creating database table...")
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")