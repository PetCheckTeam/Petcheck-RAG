import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class IngredientKnowledge(Base):
    __tablename__ = "ingredient_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    raw_name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    caution = Column(Text, nullable=True)
    embedding = Column(Vector(1024), nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()