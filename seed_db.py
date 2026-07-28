import json
import os
from database import engine, SessionLocal, Base, IngredientKnowledge
from clova_client import get_clova_embedding


def init_db():
    print("DB 테이블 생성 중...")
    Base.metadata.create_all(bind=engine)
    print("DB 테이블 생성 완료.")


def seed_data(json_file_path: str = "ingredients_seed.json"):
    if not os.path.exists(json_file_path):
        print(f"오류: '{json_file_path}' 파일을 찾을 수 없습니다.")
        return

    db = SessionLocal()
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        print(f"총 {len(data_list)}개의 성분 데이터를 삽입/업데이트합니다.")

        for item in data_list:
            raw_name = item.get("raw_name")
            canonical_name = item.get("canonical_name")
            category = item.get("category")
            description = item.get("description")
            caution = item.get("caution")

            if not raw_name or not canonical_name:
                continue

            existing = (
                db.query(IngredientKnowledge)
                .filter(IngredientKnowledge.raw_name == raw_name)
                .first()
            )

            text_to_embed = f"{raw_name} ({canonical_name}) - {description or ''}"
            embedding = get_clova_embedding(text_to_embed)

            if existing:
                existing.canonical_name = canonical_name
                existing.category = category
                existing.description = description
                existing.caution = caution
                existing.embedding = embedding
                print(f"업데이트 완료: {raw_name}")
            else:
                new_item = IngredientKnowledge(
                    raw_name=raw_name,
                    canonical_name=canonical_name,
                    category=category,
                    description=description,
                    caution=caution,
                    embedding=embedding,
                )
                db.add(new_item)
                print(f"새로 추가 완료: {raw_name}")

        db.commit()
        print("모든 데이터 시딩 작업이 완료되었습니다!")

    except Exception as e:
        db.rollback()
        print(f"시딩 중 오류 발생: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    seed_data()