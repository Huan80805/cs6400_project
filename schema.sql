-- schema.sql  (sqlite3 your.db < schema.sql)

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Reviews subset
DROP TABLE IF EXISTS reviews;
CREATE TABLE reviews (
  parent_asin       TEXT,
  rating            REAL,
  review_title      TEXT,
  review_text       TEXT,
  images_json       TEXT,   -- JSON text
  user_id           TEXT,
  review_ts         TEXT,   -- 'YYYY-MM-DD HH:MM:SS' (UTC-like / naive)
  helpful_vote      INTEGER,
  verified_purchase INTEGER, -- 0/1/NULL
  marketplace       TEXT,
  category          TEXT
);

-- Item metadata
DROP TABLE IF EXISTS products;
CREATE TABLE products (
  product_id           INTEGER PRIMARY KEY,
  parent_asin          TEXT,
  main_category        TEXT,
  product_title        TEXT,
  average_rating       REAL,
  rating_number        INTEGER,
  features_json        TEXT,   -- JSON text
  product_description  TEXT,
  price                REAL,
  images_json          TEXT,   -- JSON text
  videos_json          TEXT,   -- JSON text
  store                TEXT,
  categories_json      TEXT,   -- JSON text
  details_json         TEXT,   -- JSON text
  bought_together_json TEXT,   -- JSON text
  brand                TEXT,
  color                TEXT,
  product_locale       TEXT
);

-- ESCI
DROP TABLE IF EXISTS esci_queries;
CREATE TABLE esci_queries (
  example_id     TEXT,
  query          TEXT,
  query_id       TEXT,
  product_id     TEXT,
  product_locale TEXT,
  esci_label     TEXT,     -- 'E','S','C','I'
  small_version  INTEGER,  -- 1 if in reduced version
  large_version  INTEGER,
  split          TEXT      -- 'train' or 'test'
);

-- Indexes, can ignore?
CREATE INDEX IF NOT EXISTS idx_reviews_asin     ON reviews(parent_asin);
CREATE INDEX IF NOT EXISTS idx_reviews_ts      ON reviews(review_ts);
CREATE INDEX IF NOT EXISTS idx_reviews_user    ON reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_meta_locale     ON products(product_locale);
CREATE INDEX IF NOT EXISTS idx_meta_category   ON products(main_category);
CREATE INDEX IF NOT EXISTS idx_esci_qid       ON esci_queries(query_id);
CREATE INDEX IF NOT EXISTS idx_esci_pid       ON esci_queries(product_id);




