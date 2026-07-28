import unittest

from ingredient_extractor import extract_ingredients


class IngredientExtractorTest(unittest.TestCase):
    def test_colon_after_title(self):
        text = "사용한 원료의 명칭: 닭고기, 옥수수, 쌀, 우유"

        self.assertEqual(
            extract_ingredients(text),
            ["닭고기", "옥수수", "쌀", "우유"],
        )

    def test_no_colon_and_manufacturer_end_title(self):
        text = (
            "사용한 원료의 명칭 통옥수수,계육분,돼지지방 "
            "제조원 Hill's Pet Nutrition"
        )

        self.assertEqual(
            extract_ingredients(text),
            ["통옥수수", "계육분", "돼지지방"],
        )

    def test_commas_inside_parentheses(self):
        text = (
            "원재료명 비타민제(비타민E,나이아신),"
            "미량광물질류(황산제일철,산화아연),타우린 제조사 테스트"
        )

        self.assertEqual(
            extract_ingredients(text),
            [
                "비타민제(비타민E,나이아신)",
                "미량광물질류(황산제일철,산화아연)",
                "타우린",
            ],
        )

    def test_actual_ocr_text(self):
        text = """등록성분량 조단백질18.0%이상,조지방13.0%이상,조섬유5.0%이하,
칼슘0.45%이상,인0.41%이상,수분10.0%이하
사용한 원료의 명칭 통옥수수,계육분,돼지지방,옥수수글루텐박,
대두밀런,계란,대두박,천연향료 (닭간),밀글루텐,대두유,젖산,
아마씨,천연향료(돼지간),L-라이신,황산칼슘,어유,염화칼륨,
정제소금,염화콜린,구연산칼륨,비타민제(비타민E,나이아신),
타우린,미량광물질류힙제(황산제일철,산화아연),L-카르니틴,
d-토코페롤(혼합형),천연향료,베타카로틴
제조원 Hill's Pet Nutrition, Inc. 원산지 미국"""

        ingredients = extract_ingredients(text)

        self.assertEqual(ingredients[0], "통옥수수")
        self.assertEqual(ingredients[-1], "베타카로틴")
        self.assertFalse(any("제조원" in item for item in ingredients))
        self.assertFalse(any("원산지" in item for item in ingredients))
        self.assertFalse(
            any("사용한 원료의 명칭" in item for item in ingredients)
        )
        self.assertIn("비타민제(비타민E,나이아신)", ingredients)
        self.assertIn(
            "미량광물질류힙제(황산제일철,산화아연)",
            ingredients,
        )

    def test_title_variants(self):
        title_variants = [
            "사용한 원료의 명칭：",
            "사용한 원료의명칭",
            "사용한원료의명칭",
            "원료의 명칭:",
            "원료의 명칭",
            "원재료명:",
            "원재료명",
            "INGREDIENTS:",
            "INGREDIENTS",
        ]

        for title in title_variants:
            with self.subTest(title=title):
                self.assertEqual(
                    extract_ingredients(f"{title} 닭고기, 쌀 제조업체 테스트"),
                    ["닭고기", "쌀"],
                )

    def test_fullwidth_parentheses_and_commas(self):
        text = (
            "원재료명 혼합제（비타민제(비타민E,나이아신)，미네랄），"
            "타우린 판매원 테스트"
        )

        self.assertEqual(
            extract_ingredients(text),
            ["혼합제（비타민제(비타민E,나이아신)，미네랄）", "타우린"],
        )

    def test_title_with_newlines_and_multiple_spaces(self):
        text = "사용한\n  원료의\t명칭： 닭고기, 쌀 보관방법: 실온"

        self.assertEqual(extract_ingredients(text), ["닭고기", "쌀"])

    def test_missing_title_returns_no_ingredients(self):
        self.assertEqual(extract_ingredients("닭고기, 옥수수, 쌀"), [])

    def test_all_end_title_variants(self):
        end_titles = [
            "제조원",
            "제조사",
            "제조업체",
            "원산지",
            "제조연월일",
            "제조일자",
            "유통기한",
            "소비기한",
            "주의사항",
            "수입판매원",
            "판매원",
            "급여방법",
            "보관방법",
        ]

        for index, end_title in enumerate(end_titles):
            colon = "：" if index % 2 else ":"
            with self.subTest(end_title=end_title, colon=colon):
                self.assertEqual(
                    extract_ingredients(
                        f"원재료명 닭고기, 쌀 {end_title}{colon} 테스트"
                    ),
                    ["닭고기", "쌀"],
                )


if __name__ == "__main__":
    unittest.main()
