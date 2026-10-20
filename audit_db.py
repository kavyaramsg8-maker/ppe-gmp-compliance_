import sqlite3
import os

DB_PATH = "gmp_monitoring.db"

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

print("--- MOST RECENT 20 EVENTS ---")
rows = cursor.execute("""
    SELECT id, timestamp, track_id, zone, violation_type,
           item, confidence, snapshot_path
    FROM violations
    ORDER BY id DESC
    LIMIT 20
""").fetchall()

for row in rows:
    print(row)

print("\n--- SNAPSHOT SUMMARY BY EVENT TYPE ---")
rows = cursor.execute("""
    SELECT violation_type,
           COUNT(*),
           SUM(CASE WHEN snapshot_path IS NULL
                         OR snapshot_path = ''
                    THEN 1 ELSE 0 END)
    FROM violations
    GROUP BY violation_type
    ORDER BY violation_type
""").fetchall()

for row in rows:
    print(row)

print("\n--- MISSING REQUIRED VALUES ---")
rows = cursor.execute("""
    SELECT id, timestamp, track_id, violation_type
    FROM violations
    WHERE timestamp IS NULL
       OR track_id IS NULL
       OR violation_type IS NULL
""").fetchall()

print(rows if rows else "None")

print("\n--- SNAPSHOT PATH CHECK ---")
rows = cursor.execute("""
    SELECT id, snapshot_path
    FROM violations
    WHERE snapshot_path IS NOT NULL
      AND snapshot_path != ''
""").fetchall()

print("Records with snapshot paths:", len(rows))

for event_id, path in rows:
    print((event_id, path, os.path.exists(path)))

connection.close()