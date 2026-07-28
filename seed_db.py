from __future__ import annotations

import csv
import json
from pathlib import Path

from clova_client import get_clova_embedding
from database import (
    Base,
    IngredientKnowledge,
    SessionLocal,
    engine,
    ensure_pgvector_extension,
)


DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "petcheck_ingredient_knowledge_full.csv"
)

REQUIRED_CSV_COLUMNS = (
    "raw_name",
    "canonical_name",
    "category",
    "description",
    "caution",
    "embedding",
)


def init_db() -> None:
    ensure_pgvector_extension()
    print("DB 테이블 생성 중...")
    Base.metadata.create_all(bind=engine)
    print("DB 테이블 생성 완료.")


def parse_embedding(value: str) -> list[float] | None:
    """CSV에 벡터 값이 있으면 파싱하고, 비어 있으면 None을 반환합니다."""
    if not value or not value.strip():
        return None

    embedding = json.loads(value)
    if not isinstance(embedding, list) or len(embedding) != 1024:
        raise ValueError("embedding은 1024개 숫자로 구성된 JSON 배열이어야 합니다.")

    return [float(number) for number in embedding]


def seed_data(csv_file_path: str | Path = DEFAULT_CSV_PATH) -> None:
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        required = ", ".join(REQUIRED_CSV_COLUMNS)
        raise FileNotFoundError(
            "원료 지식 CSV 파일을 찾을 수 없습니다.\n"
            f"확인한 경로: {csv_path.resolve()}\n"
            f"필수 컬럼: {required}"
        )

    db = SessionLocal()

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            actual_columns = set(reader.fieldnames or [])
            data_list = list(reader)

        missing_columns = set(REQUIRED_CSV_COLUMNS) - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            required = ", ".join(REQUIRED_CSV_COLUMNS)
            raise ValueError(
                f"CSV 필수 컬럼이 누락되었습니다: {missing}. "
                f"파일 경로: {csv_path.resolve()}. 필수 컬럼: {required}"
            )

        print(f"총 {len(data_list)}개의 원료 데이터를 삽입/업데이트합니다.")

        for row_number, item in enumerate(data_list, start=2):
            try:
                raw_name = item["raw_name"].strip()
                canonical_name = item["canonical_name"].strip()
                category = item["category"].strip() or None
                description = item["description"].strip() or None
                caution = item["caution"].strip() or None

                if not raw_name or not canonical_name:
                    raise ValueError("raw_name과 canonical_name은 비어 있을 수 없습니다.")

                embedding = parse_embedding(item["embedding"])
                if embedding is None:
                    text_to_embed = (
                        f"원문 원료명: {raw_name}\n"
                        f"표준 원료명: {canonical_name}\n"
                        f"분류: {category or ''}\n"
                        f"설명: {description or ''}\n"
                        f"주의사항: {caution or ''}"
                    )
                    embedding = get_clova_embedding(text_to_embed)

                existing = (
                    db.query(IngredientKnowledge)
                    .filter(IngredientKnowledge.raw_name == raw_name)
                    .first()
                )

                if existing:
                    existing.canonical_name = canonical_name
                    existing.category = category
                    existing.description = description
                    existing.caution = caution
                    existing.embedding = embedding
                    print(f"업데이트 완료: {raw_name} -> {canonical_name}")
                else:
                    db.add(
                        IngredientKnowledge(
                            raw_name=raw_name,
                            canonical_name=canonical_name,
                            category=category,
                            description=description,
                            caution=caution,
                            embedding=embedding,
                        )
                    )
                    print(f"새로 추가 완료: {raw_name} -> {canonical_name}")
            except Exception as error:
                raise ValueError(
                    f"CSV {row_number}번째 줄 처리 실패: {error}"
                ) from error

        db.commit()
        print("모든 데이터 시딩 작업이 완료되었습니다!")

    except Exception as error:
        db.rollback()
        print(f"시딩 중 오류 발생: {error}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    seed_data()
