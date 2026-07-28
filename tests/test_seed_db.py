import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("CLOVA_STUDIO_API_KEY", "test-key")

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from database import Base, IngredientKnowledge
from main import RagSearchRequest, app, search_rag_context
from seed_db import (
    DEFAULT_CSV_PATH,
    EMBEDDING_DIMENSION,
    EXPECTED_STANDARD_NAMES,
    REQUIRED_CSV_COLUMNS,
    build_db_values,
    count_null_ingredient_ids,
    apply_ingredient_id_not_null,
    ensure_ingredient_id_column,
    ensure_ingredient_id_index,
    load_and_validate_csv,
    parse_args,
    seed_data,
    synchronize_id_sequence,
)


def fake_embedding(_: str) -> list[float]:
    return [0.25] * EMBEDDING_DIMENSION


class SeedDbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_rows = load_and_validate_csv(DEFAULT_CSV_PATH)

    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

    def tearDown(self):
        self.engine.dispose()

    def add_old_row(self) -> None:
        with self.session_factory() as db:
            db.add(
                IngredientKnowledge(
                    id=999,
                    ingredient_id=99,
                    raw_name="기존 원료",
                    canonical_name="기존 분류",
                    category="기존 카테고리",
                    description="기존 설명",
                    caution="기존 주의사항",
                    embedding=[0.0] * EMBEDDING_DIMENSION,
                )
            )
            db.commit()

    def run_seed(self, *, replace: bool):
        return seed_data(
            DEFAULT_CSV_PATH,
            replace=replace,
            session_factory=self.session_factory,
            embedding_provider=fake_embedding,
            database_initializer=None,
            schema_initializer=lambda _: None,
        )

    def test_csv_required_columns_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "missing_embedding.csv"
            columns = [
                column
                for column in REQUIRED_CSV_COLUMNS
                if column != "embedding"
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()

            with self.assertRaisesRegex(ValueError, "embedding"):
                load_and_validate_csv(csv_path)

    def test_csv_has_exactly_eight_ingredient_ids_and_standard_names(self):
        self.assertEqual(
            len({row.ingredient_id for row in self.csv_rows}),
            8,
        )
        self.assertEqual(
            {row.standard_name for row in self.csv_rows},
            set(EXPECTED_STANDARD_NAMES),
        )
        self.assertEqual(len(self.csv_rows), 119)

    def test_csv_to_db_field_mapping(self):
        row = self.csv_rows[0]
        values = build_db_values(row, fake_embedding)

        self.assertEqual(values["id"], row.id)
        self.assertEqual(values["ingredient_id"], row.ingredient_id)
        self.assertEqual(values["raw_name"], row.alias_name)
        self.assertEqual(values["canonical_name"], row.standard_name)
        self.assertEqual(values["description"], row.description)
        self.assertEqual(values["embedding"], fake_embedding(""))
        self.assertIsNone(values["category"])
        self.assertIsNone(values["caution"])

    def test_ingredient_id_model_column_and_existing_db_alter(self):
        column = IngredientKnowledge.__table__.c.ingredient_id
        self.assertFalse(column.nullable)
        self.assertTrue(column.index)

        db = Mock()
        ensure_ingredient_id_column(db)
        statement = " ".join(str(db.execute.call_args.args[0]).split())
        self.assertEqual(
            statement,
            "ALTER TABLE ingredient_knowledge "
            "ADD COLUMN IF NOT EXISTS ingredient_id INTEGER",
        )

    def test_ingredient_id_index_creation_sql(self):
        db = Mock()

        ensure_ingredient_id_index(db)

        statement = " ".join(str(db.execute.call_args.args[0]).split())
        self.assertEqual(
            statement,
            "CREATE INDEX IF NOT EXISTS "
            "ix_ingredient_knowledge_ingredient_id "
            "ON ingredient_knowledge (ingredient_id)",
        )

    def test_blank_embedding_uses_required_text_and_validates_dimension(self):
        captured_texts = []

        def capture_embedding(text_value: str) -> list[float]:
            captured_texts.append(text_value)
            return [0.5] * EMBEDDING_DIMENSION

        row = self.csv_rows[0]
        values = build_db_values(row, capture_embedding)

        self.assertEqual(
            captured_texts,
            [
                f"표준 원료명: {row.standard_name}\n"
                f"별칭: {row.alias_name}\n"
                f"설명: {row.description}"
            ],
        )
        self.assertEqual(len(values["embedding"]), EMBEDDING_DIMENSION)

        with self.assertRaisesRegex(ValueError, "1024차원"):
            build_db_values(row, lambda _: [0.0] * 10)

    def test_replace_removes_old_rows_and_is_idempotent(self):
        self.add_old_row()

        first_summary = self.run_seed(replace=True)

        self.assertEqual(first_summary.inserted_count, 119)
        self.assertEqual(first_summary.updated_count, 0)
        self.assertEqual(first_summary.failed_count, 0)
        self.assertEqual(first_summary.null_embedding_count, 0)

        with self.session_factory() as db:
            self.assertEqual(db.query(IngredientKnowledge).count(), 119)
            self.assertIsNone(db.get(IngredientKnowledge, 999))
            first = db.get(IngredientKnowledge, 1)
            self.assertEqual(first.ingredient_id, self.csv_rows[0].ingredient_id)
            self.assertEqual(first.raw_name, self.csv_rows[0].alias_name)
            self.assertEqual(first.canonical_name, self.csv_rows[0].standard_name)
            self.assertIsNone(first.category)
            self.assertIsNone(first.caution)

        second_summary = self.run_seed(replace=False)

        self.assertEqual(second_summary.inserted_count, 0)
        self.assertEqual(second_summary.updated_count, 119)
        with self.session_factory() as db:
            self.assertEqual(db.query(IngredientKnowledge).count(), 119)

    def test_replace_validates_nulls_before_applying_not_null(self):
        events = []
        original_null_counter = count_null_ingredient_ids

        def validate_nulls(db):
            events.append("null_validation")
            self.assertEqual(db.query(IngredientKnowledge).count(), 119)
            return original_null_counter(db)

        def apply_not_null(_db):
            events.append("not_null")
            return True

        with patch(
            "seed_db.count_null_ingredient_ids",
            side_effect=validate_nulls,
        ), patch(
            "seed_db.apply_ingredient_id_not_null",
            side_effect=apply_not_null,
        ), patch(
            "seed_db.ensure_ingredient_id_index",
        ) as create_index, patch(
            "seed_db.synchronize_id_sequence",
        ) as synchronize_sequence:
            self.run_seed(replace=True)

        self.assertEqual(events, ["null_validation", "not_null"])
        create_index.assert_called_once()
        synchronize_sequence.assert_called_once()

    def test_replace_does_not_apply_not_null_when_null_rows_remain(self):
        self.add_old_row()

        with patch(
            "seed_db.count_null_ingredient_ids",
            return_value=1,
        ), patch(
            "seed_db.apply_ingredient_id_not_null",
        ) as apply_not_null:
            with self.assertRaisesRegex(ValueError, "null인 행"):
                self.run_seed(replace=True)

        apply_not_null.assert_not_called()
        with self.session_factory() as db:
            self.assertEqual(db.query(IngredientKnowledge).count(), 1)
            self.assertIsNotNone(db.get(IngredientKnowledge, 999))

    def test_postgresql_only_schema_sql(self):
        db = Mock()
        db.get_bind.return_value = SimpleNamespace(
            dialect=SimpleNamespace(name="postgresql")
        )

        self.assertTrue(apply_ingredient_id_not_null(db))
        self.assertTrue(synchronize_id_sequence(db))

        statements = [
            " ".join(str(call.args[0]).split())
            for call in db.execute.call_args_list
        ]
        self.assertEqual(
            statements[0],
            "ALTER TABLE ingredient_knowledge "
            "ALTER COLUMN ingredient_id SET NOT NULL",
        )
        self.assertIn("SELECT setval(", statements[1])
        self.assertIn("pg_get_serial_sequence(", statements[1])
        self.assertIn("SELECT MAX(id) FROM ingredient_knowledge", statements[1])

    def test_sqlite_skips_postgresql_only_schema_sql(self):
        db = Mock()
        db.get_bind.return_value = SimpleNamespace(
            dialect=SimpleNamespace(name="sqlite")
        )

        self.assertFalse(apply_ingredient_id_not_null(db))
        self.assertFalse(synchronize_id_sequence(db))
        db.execute.assert_not_called()

    def test_failure_rolls_back_replace_and_inserts(self):
        self.add_old_row()

        def fail_flush(session, _flush_context, _instances):
            if any(item.id != 999 for item in session.new):
                raise RuntimeError("강제 flush 실패")

        event.listen(self.session_factory.class_, "before_flush", fail_flush)
        try:
            with self.assertRaisesRegex(RuntimeError, "강제 flush 실패"):
                self.run_seed(replace=True)
        finally:
            event.remove(self.session_factory.class_, "before_flush", fail_flush)

        with self.session_factory() as db:
            self.assertEqual(db.query(IngredientKnowledge).count(), 1)
            self.assertIsNotNone(db.get(IngredientKnowledge, 999))
            self.assertIsNone(db.get(IngredientKnowledge, 1))

    def test_cli_options(self):
        args = parse_args(
            [
                "--csv",
                "data/petcheck_ingredient_knowledge_8_categories.csv",
                "--replace",
            ]
        )

        self.assertEqual(args.csv.resolve(), DEFAULT_CSV_PATH.resolve())
        self.assertTrue(args.replace)

    def test_existing_rag_search_exact_match(self):
        class Match:
            canonical_name = "닭고기"
            category = None
            description = "닭고기 설명"

        class Query:
            def filter(self, *_args):
                return self

            def first(self):
                return Match()

        class FakeDb:
            def query(self, *_args):
                return Query()

        response = search_rag_context(
            RagSearchRequest(
                analysisId=1,
                ocrText="원재료명 닭고기 제조원 테스트",
                topK=1,
            ),
            db=FakeDb(),
        )

        self.assertEqual(response.extractedIngredients, ["닭고기"])
        self.assertEqual(response.contexts[0].ingredientName, "닭고기")
        self.assertTrue(
            any(route.path == "/api/v1/rag/search" for route in app.routes)
        )


if __name__ == "__main__":
    unittest.main()
