
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "gmp_monitoring.db"

results = []


def report(name, passed, details=""):
    status = "PASS" if passed else "CHECK"
    results.append((name, passed))
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


def check_files():
    print("\n=== 1. PROJECT FILES ===")

    required = [
        "main.py",
        "replay.py",
        "database/logger.py",
        "rule_engine/smoothing.py",
        "rule_engine/snapshots.py",
        "dashboard/app.py",
        "test_gowning_sequence.py",
        "test_manager_logging_failure.py",
        "test_ml_adapter.py",
        "test_replay.py",
        "test_zones_integration.py",
    ]

    for filename in required:
        exists = (ROOT / filename).is_file()
        report(
            f"File: {filename}",
            exists,
            "Found" if exists else "Missing or filename differs"
        )


def check_tests():
    print("\n=== 2. AUTOMATED TESTS ===")

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

        output = (completed.stdout + "\n" + completed.stderr).strip()
        print(output[-4000:] if output else "(No test output)")

        report(
            "Full pytest suite",
            completed.returncode == 0,
            "Tests passed" if completed.returncode == 0
            else "Investigate failing tests"
        )

    except subprocess.TimeoutExpired:
        report("Full pytest suite", False, "Timed out after 180 seconds")
    except Exception as error:
        report("Full pytest suite", False, str(error))


def check_database():
    print("\n=== 3. DATABASE ===")

    if not DB_PATH.is_file():
        report("Database exists", False, str(DB_PATH))
        return

    report("Database exists", True, DB_PATH.name)

    connection = None
    try:
        # Read-only connection: this script does not change the database.
        uri = DB_PATH.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        report(
            "SQLite integrity",
            integrity == "ok",
            integrity
        )

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        has_table = "violations" in tables
        report("Violations table", has_table)

        if has_table:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(violations)"
                )
            }

            expected = {
                "id", "timestamp", "track_id", "zone",
                "violation_type", "item", "confidence",
                "snapshot_path"
            }

            schema_ok = expected.issubset(columns)
            report(
                "Violation table schema",
                schema_ok,
                "Required columns present" if schema_ok
                else f"Missing columns: {sorted(expected - columns)}"
            )

            count = connection.execute(
                "SELECT COUNT(*) FROM violations"
            ).fetchone()[0]

            latest = connection.execute(
                "SELECT MAX(id) FROM violations"
            ).fetchone()[0]

            print(f"       Total events: {count}")
            print(f"       Latest event ID: {latest}")

            report("Database contains events", count > 0)

            types = connection.execute("""
                SELECT violation_type, COUNT(*)
                FROM violations
                GROUP BY violation_type
                ORDER BY violation_type
            """).fetchall()

            print("       Event types:")
            for event_type, event_count in types:
                print(f"         {event_type}: {event_count}")

            snapshot_rows = connection.execute("""
                SELECT snapshot_path
                FROM violations
                WHERE snapshot_path IS NOT NULL
                  AND TRIM(snapshot_path) != ''
            """).fetchall()

            existing = 0
            missing = []

            for (stored_path,) in snapshot_rows:
                path = Path(stored_path)
                if not path.is_absolute():
                    path = ROOT / path

                if path.is_file():
                    existing += 1
                else:
                    missing.append(stored_path)

            report(
                "Stored snapshot paths",
                not missing,
                f"{existing} existing; {len(missing)} missing"
            )

            if missing:
                for path in missing[:5]:
                    print(f"         Missing: {path}")

    except Exception as error:
        report("Database audit", False, str(error))
    finally:
        if connection is not None:
            connection.close()


def check_demo_materials():
    print("\n=== 4. DEMO AND PRESENTATION MATERIALS ===")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    document_extensions = {".pptx", ".pdf", ".docx"}

    videos = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in video_extensions
        and not any(part in {"venv", ".git", "__pycache__"}
                    for part in p.parts)
    ]

    documents = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in document_extensions
        and not any(part in {"venv", ".git", "__pycache__"}
                    for part in p.parts)
    ]

    report(
        "Demo video available",
        bool(videos),
        f"{len(videos)} video file(s) found"
    )

    report(
        "Presentation or documentation file",
        bool(documents),
        f"{len(documents)} PDF/PPTX/DOCX file(s) found"
    )


def main():
    print("=" * 60)
    print("GMP PPE PRESENTATION READINESS AUDIT")
    print(f"Project: {ROOT}")
    print("=" * 60)

    check_files()
    check_tests()
    check_database()
    check_demo_materials()

    print("\n=== 5. MANUAL VERIFICATION REQUIRED ===")
    manual_checks = [
        "Launch the actual application and confirm video input works.",
        "Verify person tracking and PPE detections on real footage.",
        "Trigger a known violation and confirm it reaches the dashboard.",
        "Open a saved snapshot and verify the image is correct.",
        "Test correct PPE, missing PPE, gowning order and zone changes.",
        "Prepare architecture slides, results, limitations and demo script.",
    ]

    for item in manual_checks:
        print(f"[MANUAL] {item}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"AUTOMATED CHECKS: {passed}/{total} passed")
    print("This score is NOT a project-completion percentage.")

    if passed == total:
        print("Automated checks look good; complete the manual checks above.")
    else:
        print("Resolve CHECK items before treating the project as ready.")

    print("This audit does not certify detection accuracy or GMP compliance.")
    print("=" * 60)


if __name__ == "__main__":
    main()
