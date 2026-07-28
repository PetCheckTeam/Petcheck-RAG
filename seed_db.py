from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from sqlalchemy import func, text
from sqlalchemy.orm import Session

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
    / "petcheck_ingredient_knowledge_8_categories.csv"
)

REQUIRED_CSV_COLUMNS = (
    "id",
    "ingredient_id",
    "standard_name",
    "alias_name",
    "description",
    "embedding",
)
EXPECTED_ROW_COUNT = 119
EXPECTED_STANDARD_NAMES = (
    "닭고기",
    "소고기",
    "돼지고기",
    "밀",
    "옥수수",
    "우유",
    "계란",
    "생선",
)
EXPECTED_INGREDIENT_ID_COUNT = 8
EMBEDDING_DIMENSION = 1024

ADD_INGREDIENT_ID_COLUMN_SQL = """
ALTER TABLE ingredient_knowledge
ADD COLUMN IF NOT EXISTS ingredient_id INTEGER
"""

CREATE_INGREDIENT_ID_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS
ix_ingredient_knowledge_ingredient_id
ON ingredient_knowledge (ingredient_id)
"""

SET_INGREDIENT_ID_NOT_NULL_SQL = """
ALTER TABLE ingredient_knowledge
ALTER COLUMN ingredient_id SET NOT NULL
"""

SYNC_ID_SEQUENCE_SQL = """
SELECT setval(
    pg_get_serial_sequence(
        'ingredient_knowledge',
        'id'
    ),
    COALESCE(
        (SELECT MAX(id) FROM ingredient_knowledge),
        1
    ),
    true
)
"""


@dataclass(frozen=True)
class CsvIngredient:
    row_number: int
    id: int
    ingredient_id: int
    standard_name: str
    alias_name: str
    description: str
    embedding: list[float] | None


@dataclass(frozen=True)
class SeedSummary:
    inserted_count: int
    updated_count: int
    failed_count: int
    ingredient_id_count: int
    standard_name_counts: dict[str, int]
    null_embedding_count: int


def init_db() -> None:
    ensure_pgvector_extension()
    print("DB 테이블 생성 중...")
    Base.metadata.create_all(bind=engine)
    print("DB 테이블 생성 완료.")


def validate_embedding(
    embedding: Sequence[object],
    source: str,
) -> list[float]:
    if not isinstance(embedding, (list, tuple)):
        raise ValueError(f"{source}은 숫자 배열이어야 합니다.")
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{source}은 {EMBEDDING_DIMENSION}차원이어야 합니다. "
            f"현재 차원: {len(embedding)}"
        )

    try:
        return [float(number) for number in embedding]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}에 숫자가 아닌 값이 포함되어 있습니다.") from error


def parse_embedding(value: str) -> list[float] | None:
    """CSV embedding을 파싱하고 빈 값이면 생성 대상으로 표시합니다."""
    if not value or not value.strip():
        return None

    try:
        embedding = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("CSV embedding은 JSON 배열이어야 합니다.") from error

    return validate_embedding(embedding, "CSV embedding")


def _parse_integer(value: str | None, column: str, row_number: int) -> int:
    try:
        return int((value or "").strip())
    except ValueError as error:
        raise ValueError(
            f"CSV {row_number}번째 줄의 {column}은 정수여야 합니다."
        ) from error


def load_and_validate_csv(csv_file_path: str | Path) -> list[CsvIngredient]:
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"원료 지식 CSV 파일을 찾을 수 없습니다: {csv_path.resolve()}"
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        actual_columns = set(reader.fieldnames or [])
        missing_columns = set(REQUIRED_CSV_COLUMNS) - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV 필수 컬럼이 누락되었습니다: {missing}")
        raw_rows = list(reader)

    if len(raw_rows) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"CSV 행 수는 정확히 {EXPECTED_ROW_COUNT}개여야 합니다. "
            f"현재 행 수: {len(raw_rows)}"
        )

    rows: list[CsvIngredient] = []
    seen_ids: set[int] = set()
    seen_aliases: set[str] = set()

    for row_number, raw_row in enumerate(raw_rows, start=2):
        row_id = _parse_integer(raw_row.get("id"), "id", row_number)
        ingredient_id = _parse_integer(
            raw_row.get("ingredient_id"),
            "ingredient_id",
            row_number,
        )
        standard_name = (raw_row.get("standard_name") or "").strip()
        alias_name = (raw_row.get("alias_name") or "").strip()
        description = (raw_row.get("description") or "").strip()

        if not standard_name:
            raise ValueError(
                f"CSV {row_number}번째 줄의 standard_name은 비어 있을 수 없습니다."
            )
        if not alias_name:
            raise ValueError(
                f"CSV {row_number}번째 줄의 alias_name은 비어 있을 수 없습니다."
            )
        if row_id in seen_ids:
            raise ValueError(f"CSV id가 중복되었습니다: {row_id}")

        alias_key = alias_name.casefold()
        if alias_key in seen_aliases:
            raise ValueError(f"CSV alias_name이 중복되었습니다: {alias_name}")

        try:
            embedding = parse_embedding(raw_row.get("embedding") or "")
        except ValueError as error:
            raise ValueError(f"CSV {row_number}번째 줄 처리 실패: {error}") from error

        seen_ids.add(row_id)
        seen_aliases.add(alias_key)
        rows.append(
            CsvIngredient(
                row_number=row_number,
                id=row_id,
                ingredient_id=ingredient_id,
                standard_name=standard_name,
                alias_name=alias_name,
                description=description,
                embedding=embedding,
            )
        )

    ingredient_ids = {row.ingredient_id for row in rows}
    if len(ingredient_ids) != EXPECTED_INGREDIENT_ID_COUNT:
        raise ValueError(
            "ingredient_id 고유값은 정확히 "
            f"{EXPECTED_INGREDIENT_ID_COUNT}개여야 합니다. "
            f"현재 개수: {len(ingredient_ids)}"
        )

    standard_names = {row.standard_name for row in rows}
    expected_standard_names = set(EXPECTED_STANDARD_NAMES)
    if standard_names != expected_standard_names:
        missing = sorted(expected_standard_names - standard_names)
        unexpected = sorted(standard_names - expected_standard_names)
        raise ValueError(
            "standard_name 고유값이 지정된 8개와 일치하지 않습니다. "
            f"누락: {missing}, 예상 외: {unexpected}"
        )

    return rows


def build_embedding_text(row: CsvIngredient) -> str:
    return (
        f"표준 원료명: {row.standard_name}\n"
        f"별칭: {row.alias_name}\n"
        f"설명: {row.description}"
    )


def build_db_values(
    row: CsvIngredient,
    embedding_provider: Callable[[str], list[float]],
) -> dict[str, object]:
    embedding = row.embedding
    if embedding is None:
        embedding = validate_embedding(
            embedding_provider(build_embedding_text(row)),
            "CLOVA embedding",
        )

    return {
        "id": row.id,
        "ingredient_id": row.ingredient_id,
        "raw_name": row.alias_name,
        "canonical_name": row.standard_name,
        "category": None,
        "description": row.description or None,
        "caution": None,
        "embedding": embedding,
    }


def ensure_ingredient_id_column(db: Session) -> None:
    db.execute(text(ADD_INGREDIENT_ID_COLUMN_SQL))


def ensure_ingredient_id_index(db: Session) -> None:
    db.execute(text(CREATE_INGREDIENT_ID_INDEX_SQL))


def count_null_ingredient_ids(db: Session) -> int:
    return int(
        db.query(func.count(IngredientKnowledge.id))
        .filter(IngredientKnowledge.ingredient_id.is_(None))
        .scalar()
        or 0
    )


def is_postgresql(db: Session) -> bool:
    return db.get_bind().dialect.name == "postgresql"


def apply_ingredient_id_not_null(db: Session) -> bool:
    if not is_postgresql(db):
        return False

    db.execute(text(SET_INGREDIENT_ID_NOT_NULL_SQL))
    return True


def synchronize_id_sequence(db: Session) -> bool:
    if not is_postgresql(db):
        return False

    db.execute(text(SYNC_ID_SEQUENCE_SQL))
    return True


def _print_summary(summary: SeedSummary) -> None:
    print(f"삽입 행 수: {summary.inserted_count}")
    print(f"수정 행 수: {summary.updated_count}")
    print(f"실패 행 수: {summary.failed_count}")
    print(f"ingredient_id 고유 개수: {summary.ingredient_id_count}")
    print("standard_name별 행 개수:")
    for standard_name in EXPECTED_STANDARD_NAMES:
        print(f"  {standard_name}: {summary.standard_name_counts[standard_name]}")
    print(f"embedding이 null인 행 개수: {summary.null_embedding_count}")


def seed_data(
    csv_file_path: str | Path = DEFAULT_CSV_PATH,
    *,
    replace: bool = False,
    session_factory: Callable[[], Session] = SessionLocal,
    embedding_provider: Callable[[str], list[float]] = get_clova_embedding,
    database_initializer: Callable[[], None] | None = init_db,
    schema_initializer: Callable[[Session], None] = ensure_ingredient_id_column,
) -> SeedSummary:
    rows = load_and_validate_csv(csv_file_path)
    print(f"CSV 검증 완료: {len(rows)}행")

    if database_initializer is not None:
        database_initializer()

    prepared_rows = []
    for row in rows:
        try:
            prepared_rows.append(build_db_values(row, embedding_provider))
        except Exception as error:
            raise ValueError(
                f"CSV {row.row_number}번째 줄 임베딩 처리 실패: {error}"
            ) from error

    db = session_factory()
    inserted_count = 0
    updated_count = 0

    try:
        with db.begin():
            schema_initializer(db)

            if replace:
                db.query(IngredientKnowledge).delete(synchronize_session=False)

            for values in prepared_rows:
                row_id = int(values["id"])
                existing = db.get(IngredientKnowledge, row_id)

                if existing is None:
                    db.add(IngredientKnowledge(**values))
                    inserted_count += 1
                    continue

                for field, value in values.items():
                    if field != "id":
                        setattr(existing, field, value)
                updated_count += 1

            db.flush()

            if replace:
                null_ingredient_id_count = count_null_ingredient_ids(db)
                if null_ingredient_id_count:
                    raise ValueError(
                        "ingredient_id가 null인 행이 남아 있습니다: "
                        f"{null_ingredient_id_count}개"
                    )
                apply_ingredient_id_not_null(db)

            ensure_ingredient_id_index(db)
            synchronize_id_sequence(db)

            null_embedding_count = int(
                db.query(func.count(IngredientKnowledge.id))
                .filter(IngredientKnowledge.embedding.is_(None))
                .scalar()
                or 0
            )
    except Exception as error:
        db.rollback()
        print("실패 행 수: 1")
        print(f"시딩 중 오류 발생, 전체 rollback 완료: {error}")
        raise
    finally:
        db.close()

    standard_name_counts = dict(
        Counter(row.standard_name for row in rows)
    )
    summary = SeedSummary(
        inserted_count=inserted_count,
        updated_count=updated_count,
        failed_count=0,
        ingredient_id_count=len({row.ingredient_id for row in rows}),
        standard_name_counts=standard_name_counts,
        null_embedding_count=null_embedding_count,
    )
    _print_summary(summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PetCheck 원료 지식 CSV 시딩")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"시딩할 CSV 경로 (기본값: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="기존 ingredient_knowledge 전체 데이터를 삭제한 뒤 시딩",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seed_data(args.csv, replace=args.replace)


if __name__ == "__main__":
    main()
