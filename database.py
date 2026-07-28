import os
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Text, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
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
    ingredient_id = Column(Integer, nullable=False, index=True)
    raw_name = Column(String(255), nullable=False, index=True)
    canonical_name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    caution = Column(Text, nullable=True)
    embedding = Column(Vector(1024), nullable=True)


def ensure_pgvector_extension() -> None:
    """Fail fast with an actionable message when pgvector is unavailable."""
    try:
        with engine.connect() as connection:
            installed = connection.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_extension
                        WHERE extname = 'vector'
                    )
                    """
                )
            ).scalar_one()
    except SQLAlchemyError as error:
        raise RuntimeError(
            "PostgreSQL 연결 또는 pgvector 확장 확인에 실패했습니다. "
            "DATABASE_URL과 데이터베이스 접속 상태를 확인하세요."
        ) from error

    if not installed:
        raise RuntimeError(
            "PostgreSQL에 pgvector 확장이 활성화되어 있지 않습니다. "
            "권한이 있는 계정으로 'CREATE EXTENSION vector;'를 실행한 뒤 다시 시도하세요."
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
