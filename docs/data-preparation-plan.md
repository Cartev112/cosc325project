# Initial implementation

1. Set up Python 3.12 with a locked environment and ignore local data and environments.
2. Download only the labeled JSON from the named Kaggle dataset. Record the URL and source checksum; allow a manual download if access requires login.
3. Validate IDs, cuisine labels, and ingredient strings. Stop on malformed data rather than silently dropping it. Normalize case and whitespace while preserving whole ingredient phrases.
4. Group recipes by their sorted, unique normalized ingredients. Keep the lowest-ID recipe in same-label groups. Exclude every recipe in conflicting-label groups and record excluded IDs and reasons locally.
5. Split the remaining unique recipes into approximately 70% train, 15% validation, and 15% test with stratification and seed 325. Check that IDs and ingredient signatures do not overlap and every cuisine appears in each split.
6. Save plain JSON records, split IDs, provenance, and the exclusion log. Generate a compact audit report and training-only plots. Refuse to replace existing prepared splits.
7. Test validation, phrase preservation, duplicate policy, deterministic splits, and leakage prevention. Run on the real data, then update the README with commands and measured counts.

Done means the environment can be reproduced and the three partitions can be loaded. Feature fitting and model training are subsequent work.
