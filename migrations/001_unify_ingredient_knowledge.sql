BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF to_regclass('public.ingredient_knowledge') IS NOT NULL THEN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'ingredient_knowledge'
              AND column_name = 'alias_name'
        ) AND NOT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'ingredient_knowledge'
              AND column_name = 'raw_name'
        ) THEN
            ALTER TABLE ingredient_knowledge RENAME COLUMN alias_name TO raw_name;
        END IF;

        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'ingredient_knowledge'
              AND column_name = 'standard_name'
        ) AND NOT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'ingredient_knowledge'
              AND column_name = 'canonical_name'
        ) THEN
            ALTER TABLE ingredient_knowledge RENAME COLUMN standard_name TO canonical_name;
        END IF;

        ALTER TABLE ingredient_knowledge
            ADD COLUMN IF NOT EXISTS category VARCHAR(100),
            ADD COLUMN IF NOT EXISTS caution TEXT;

        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'ingredient_knowledge'
              AND column_name = 'ingredient_id'
        ) THEN
            ALTER TABLE ingredient_knowledge DROP COLUMN ingredient_id;
        END IF;

        ALTER TABLE ingredient_knowledge
            ALTER COLUMN embedding TYPE vector(1024);
    END IF;
END
$$;

COMMIT;
