-- Creates the synthetic click-through-rate table that train.py reads: 1 million rows of user and item
-- features with a click label. About 80% of rows have split = 'train' and the rest have split = 'test'.
-- To create the table somewhere else, change the name here and parameters.table in workload.yaml.
CREATE TABLE IF NOT EXISTS main.default.air_examples_ctr_features AS
WITH draws AS (
  SELECT
    id,
    CAST(18 + FLOOR(rand(1) * 50) AS INT) AS user_age,
    CAST(FLOOR(rand(2) * 3650) AS INT) AS user_tenure_days,
    ROUND(5 + rand(3) * 195, 2) AS item_price,
    ROUND(rand(4) * 0.2, 4) AS user_ctr_7d,
    CAST(FLOOR(rand(5) * 100000) AS INT) AS item_views_7d,
    rand(6) AS split_draw,
    rand(7) AS click_draw
  FROM range(1000000)
)
SELECT
  id,
  user_age,
  user_tenure_days,
  item_price,
  user_ctr_7d,
  item_views_7d,
  IF(split_draw < 0.8, 'train', 'test') AS split,
  -- Clicks are more likely for users who clicked recently and for popular, cheaper items.
  CAST(click_draw < 1 / (1 + exp(2 - 12 * user_ctr_7d - 1.5 * item_views_7d / 100000 + 0.01 * item_price)) AS INT) AS label
FROM draws;
