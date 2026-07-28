import csv
import json
from pathlib import Path

from sqlalchemy import text

from clova_client import get_clova_embedding
from database import Base, IngredientKnowledge, SessionLocal, engine


DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "petcheck_ingredient_knowledge_full.csv"
)


def init_db() -> None:
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
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        return

    db = SessionLocal()

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            data_list = list(csv.DictReader(csv_file))

        required_columns = {
            "id",
            "ingredient_id",
            "standard_name",
            "alias_name",
            "description",
            "embedding",
        }
        actual_columns = set(data_list[0].keys()) if data_list else set()
        missing_columns = required_columns - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")

        print(f"총 {len(data_list)}개의 원료 데이터를 삽입/업데이트합니다.")

        for row_number, item in enumerate(data_list, start=2):
            try:
                item_id = int(item["id"])
                ingredient_id = int(item["ingredient_id"])
                standard_name = item["standard_name"].strip()
                alias_name = item["alias_name"].strip()
                description = item["description"].strip() or None

                if not standard_name or not alias_name:
                    raise ValueError("standard_name과 alias_name은 비어 있을 수 없습니다.")

                embedding = parse_embedding(item["embedding"])
                if embedding is None:
                    text_to_embed = (
                        f"표준 원료명: {standard_name}\n"
                        f"별칭: {alias_name}\n"
                        f"설명: {description or ''}"
                    )
                    embedding = get_clova_embedding(text_to_embed)

                existing = db.get(IngredientKnowledge, item_id)

                if existing:
                    existing.ingredient_id = ingredient_id
                    existing.standard_name = standard_name
                    existing.alias_name = alias_name
                    existing.description = description
                    existing.embedding = embedding
                    print(f"업데이트 완료: {alias_name} -> {standard_name}")
                else:
                    db.add(
                        IngredientKnowledge(
                            id=item_id,
                            ingredient_id=ingredient_id,
                            standard_name=standard_name,
                            alias_name=alias_name,
                            description=description,
                            embedding=embedding,
                        )
                    )
                    print(f"새로 추가 완료: {alias_name} -> {standard_name}")
            except Exception as error:
                raise ValueError(
                    f"CSV {row_number}번째 줄 처리 실패: {error}"
                ) from error

        # CSV의 id를 직접 저장했으므로 PostgreSQL 자동 증가 시퀀스를 맞춥니다.
        db.flush()
        db.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence('ingredient_knowledge', 'id'),
                    COALESCE((SELECT MAX(id) FROM ingredient_knowledge), 1),
                    true
                )
                """
            )
        )
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
