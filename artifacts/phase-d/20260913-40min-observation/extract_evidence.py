"""Create a read-only, reproducible evidence bundle for one product session."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


SESSION_ID = "aed4a2ef-2915-46c7-a04b-da1a4addbaf7"
LOCAL_ZONE = ZoneInfo("America/Denver")
OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parents[2]
DATABASE = PROJECT_ROOT / "data" / "product" / "observer.sqlite3"


def rows(connection: sqlite3.Connection, sql: str, parameters=()) -> list[dict]:
    return [dict(row) for row in connection.execute(sql, parameters)]


def write_json(name: str, value: object) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(name: str, fieldnames: list[str], values: list[dict]) -> None:
    with (OUTPUT_DIR / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def local_time(started: datetime, offset: float) -> str:
    return (started + timedelta(seconds=offset)).astimezone(LOCAL_ZONE).isoformat(
        timespec="milliseconds"
    )


def main() -> None:
    connection = sqlite3.connect(
        f"file:{DATABASE.as_posix()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    session_rows = rows(connection, "SELECT * FROM sessions WHERE id=?", (SESSION_ID,))
    if len(session_rows) != 1:
        raise RuntimeError(f"expected one session, found {len(session_rows)}")
    session = session_rows[0]
    task_rows = rows(connection, "SELECT * FROM tasks WHERE id=?", (session["task_id"],))
    totals = rows(
        connection,
        "SELECT * FROM totals WHERE session_id=? ORDER BY signal",
        (SESSION_ID,),
    )
    actions = rows(
        connection,
        "SELECT * FROM actions WHERE session_id=? ORDER BY offset,rowid",
        (SESSION_ID,),
    )
    evidence = rows(
        connection, "SELECT * FROM evidence WHERE session_id=?", (SESSION_ID,)
    )
    settings = rows(connection, "SELECT key,value FROM settings ORDER BY key")
    connection.rollback()
    connection.close()

    parsed_session = dict(session)
    for key in ("identity", "policy", "resource_summary"):
        parsed_session[key] = json.loads(parsed_session[key]) if parsed_session[key] else None
    bundle = {
        "source_database": str(DATABASE),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": parsed_session,
        "task": task_rows[0] if task_rows else None,
        "totals": totals,
        "actions": actions,
        "evidence": evidence,
        "settings": settings,
    }
    write_json("raw-session-records.json", bundle)
    write_json("runtime-identity-and-policy.json", {
        "identity": parsed_session["identity"],
        "policy": parsed_session["policy"],
    })
    write_json("resource-summary.json", parsed_session["resource_summary"])

    signal_rows = []
    for total in totals:
        eligible = float(total["eligible"])
        valid = float(total["valid"])
        violation = float(total["violation"])
        signal_rows.append({
            "signal": total["signal"],
            "eligible_seconds": f"{eligible:.3f}",
            "valid_seconds": f"{valid:.3f}",
            "unknown_seconds": f"{eligible-valid:.3f}",
            "valid_coverage_percent": f"{100*valid/eligible:.3f}" if eligible else "",
            "violation_seconds": f"{violation:.3f}",
            "violation_of_valid_percent": f"{100*violation/valid:.3f}" if valid else "",
        })
    write_csv(
        "signal-totals.csv",
        ["signal", "eligible_seconds", "valid_seconds", "unknown_seconds",
         "valid_coverage_percent", "violation_seconds",
         "violation_of_valid_percent"],
        signal_rows,
    )

    started = datetime.fromisoformat(session["started"])
    timeline_rows = []
    for action in actions:
        timeline_rows.append({
            "offset_seconds": f"{float(action['offset']):.3f}",
            "local_time": local_time(started, float(action["offset"])),
            "rule_id": action["rule_id"],
            "kind": action["kind"],
            "points": f"{float(action['points']):.1f}",
            "reason": action["reason"] or "",
            "delivery_state": action["delivery_state"],
            "delivery_attempted_utc": action["delivery_attempted"] or "",
            "delivery_error": action["delivery_error"] or "",
            "event_id": action["event_id"],
            "action_id": action["id"],
        })
    write_csv(
        "event-timeline.csv",
        ["offset_seconds", "local_time", "rule_id", "kind", "points", "reason",
         "delivery_state", "delivery_attempted_utc", "delivery_error", "event_id",
         "action_id"],
        timeline_rows,
    )

    totals_by_signal = {item["signal"]: item for item in totals}
    hard_coverage = min(
        float(totals_by_signal[name]["valid"]) /
        float(totals_by_signal[name]["eligible"])
        for name in ("away", "phone_held")
    )
    alerts = [item for item in actions if item["kind"] == "alert"]
    penalties = [item for item in actions if item["kind"] == "penalty"]
    resource = parsed_session["resource_summary"]
    checks = {
        "session_id": SESSION_ID,
        "task_id": session["task_id"],
        "session_state": session["state"],
        "task_state": task_rows[0]["state"] if task_rows else None,
        "elapsed_seconds_from_totals": float(totals_by_signal["away"]["eligible"]),
        "hard_signal_coverage_percent": 100 * hard_coverage,
        "score": max(0.0, 100.0 - sum(float(item["points"]) for item in actions)),
        "deducted_points": sum(float(item["points"]) for item in penalties),
        "alert_count": len(alerts),
        "alert_count_by_rule": dict(Counter(item["rule_id"] for item in alerts)),
        "delivered_alert_count": sum(
            item["delivery_state"] == "delivered" for item in alerts
        ),
        "delivery_error_count": sum(bool(item["delivery_error"]) for item in alerts),
        "evidence_image_count": len(evidence),
        "resource_within_budget": resource["within_budget"] if resource else None,
        "resource_violation_count": len(resource["violations"]) if resource else None,
    }
    write_json("verification-checks.json", checks)

    queries = f"""-- Run against data/product/observer.sqlite3
.headers on
.mode box
SELECT * FROM sessions WHERE id='{SESSION_ID}';
SELECT * FROM tasks WHERE id='{session['task_id']}';
SELECT * FROM totals WHERE session_id='{SESSION_ID}' ORDER BY signal;
SELECT offset,rule_id,kind,points,reason,delivery_state,delivery_attempted,delivery_error
FROM actions WHERE session_id='{SESSION_ID}' ORDER BY offset,rowid;
SELECT * FROM evidence WHERE session_id='{SESSION_ID}';
"""
    (OUTPUT_DIR / "reproduce-queries.sql").write_text(queries, encoding="utf-8")

    generated = [
        "README.md",
        "raw-session-records.json",
        "runtime-identity-and-policy.json",
        "resource-summary.json",
        "signal-totals.csv",
        "event-timeline.csv",
        "verification-checks.json",
        "reproduce-queries.sql",
        "extract_evidence.py",
    ]
    manifest_lines = []
    for name in generated:
        digest = hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {name}")
    (OUTPUT_DIR / "SHA256SUMS.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
