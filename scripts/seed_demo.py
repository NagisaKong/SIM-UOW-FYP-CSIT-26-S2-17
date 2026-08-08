"""Seed the demo database: build the roster, hash every password, run the SQL.

Why this exists rather than `psql -f database/seed_demo.sql`:

Every account now has its OWN password (a student's password is their student
ID), and Argon2id cannot be computed inside PostgreSQL. So the roster and its
hashes are built here, staged into a TEMPORARY table on the same connection,
and `database/seed_demo.sql` reads the accounts back out of it. The SQL file
still owns the whole demo scenario — courses, sessions, attendance, appeals,
leave, behaviour — this script only supplies the people.

    python scripts/seed_demo.py            # dry run: show what would happen
    python scripts/seed_demo.py --yes      # actually wipe and re-seed

DESTRUCTIVE: the SQL deletes every row of demo data before re-inserting, so
never point it at anything you are not willing to lose.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import psycopg2
from argon2 import PasswordHasher, Type
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_SQL = _REPO_ROOT / "database" / "seed_demo.sql"

# Same parameters as core/userInformation.py — a hash produced here has to
# verify against the login endpoint, so these must not drift apart.
_HASHER = PasswordHasher(type=Type.ID, memory_cost=65536, time_cost=3, parallelism=4)

ADMIN_PASSWORD = "demo123"
COHORT_SIZE = 200          # total student accounts, the 5 real ones included
EMAIL_DOMAIN = "demo.com"  # fabricated accounts only; the team keep their own


@dataclass
class Account:
    demo_key: str          # stable handle the SQL joins on
    role: str              # student | teacher | admin
    email: str
    full_name: str
    student_id: str | None = None
    staff_id: str | None = None
    status: str = "active"
    password: str = ""     # plaintext; hashed before it reaches the database
    sort_key: int = 0
    password_hash: str = field(default="", repr=False)


# ── The five real team members ───────────────────────────────────────
# Their own SIM addresses and UOW student numbers, so each of them can log in
# as themselves during the presentation. Password = their student number.
# Face data is deliberately NOT seeded for them — they enrol it themselves.
TEAM = [
    ("team_dominic",  "WHYE LI HENG, DOMINIC", "9891092", "Lhdwhye001@mymail.sim.edu.sg"),
    ("team_zhanghao", "YU, ZHANGHAO",          "9107071", "zyu010@mymail.sim.edu.sg"),
    ("team_chengwei", "ZHANG, CHENGWEI",       "9182858", "czhang029@mymail.sim.edu.sg"),
    ("team_jiqian",   "ZHANG, JIQIAN",         "8466907", "jzhang092@mymail.sim.edu.sg"),
    ("team_shiyin",   "ZHAO, SHIYIN",          "9107356", "szhao015@mymail.sim.edu.sg"),
]

# Name pools for the fabricated cohort. Kept plausible for a SIM/UOW intake so
# the demo does not look like it is full of placeholder text.
_SURNAMES = [
    "Tan", "Lim", "Lee", "Ng", "Wong", "Chan", "Koh", "Goh", "Teo", "Ong",
    "Chua", "Yeo", "Sim", "Toh", "Low", "Neo", "Seah", "Foo", "Heng", "Quek",
    "Kumar", "Raj", "Devi", "Menon", "Pillai", "Nair", "Shah", "Patel",
    "Zhang", "Wang", "Li", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou",
    "Nguyen", "Tran", "Pham", "Vo", "Do", "Hoang", "Bui", "Dang",
]
_GIVEN = [
    "Wei Ming", "Jia Hui", "Zhi Hao", "Xin Yi", "Jun Kai", "Mei Ling",
    "Yong Sheng", "Hui Fen", "Kai Xuan", "Shu Ting", "Wen Jie", "Li Ying",
    "Aarav", "Divya", "Rohan", "Priya", "Arjun", "Kavya", "Vikram", "Ananya",
    "Daniel", "Rachel", "Marcus", "Chloe", "Ethan", "Natalie", "Ryan", "Grace",
    "Minh Anh", "Duc Anh", "Thu Ha", "Quang Huy",
]


def build_roster() -> list[Account]:
    """Deterministic roster: same input, same accounts, every single run."""
    accounts: list[Account] = [
        Account("admin", "admin", "admin@demo.local", "Demo Admin",
                staff_id="A00001", password=ADMIN_PASSWORD, sort_key=0),
        Account("teacher_tara", "teacher", f"twong@{EMAIL_DOMAIN}", "Tara Wong",
                staff_id="T00001", password="T00001", sort_key=1),
        Account("teacher_tom", "teacher", f"tchen@{EMAIL_DOMAIN}", "Tom Chen",
                staff_id="T00002", password="T00002", sort_key=2),
        Account("teacher_ian", "teacher", f"ilee@{EMAIL_DOMAIN}", "Ian Lee",
                staff_id="T00003", password="T00003", status="inactive",
                sort_key=3),
    ]

    # The team first, so they land on the lowest student ids.
    for i, (key, name, uow_id, email) in enumerate(TEAM):
        accounts.append(Account(
            demo_key=key, role="student", email=email, full_name=name,
            student_id=uow_id, password=uow_id, sort_key=10 + i,
        ))

    # …then the fabricated cohort, mirroring the same shape: a SIM-style login
    # built from the name, a 7-digit student number, password = that number.
    seen_emails = {a.email.lower() for a in accounts}
    for i in range(COHORT_SIZE - len(TEAM)):
        given = _GIVEN[i % len(_GIVEN)]
        surname = _SURNAMES[(i * 7 + i // len(_SURNAMES)) % len(_SURNAMES)]
        student_id = str(8000000 + i * 137 % 2000000).zfill(7)
        local = f"{given[0]}{surname}{i:03d}".lower()
        email = f"{local}@{EMAIL_DOMAIN}"
        if email in seen_emails:            # belt and braces; local part has i
            email = f"{local}x{i}@{EMAIL_DOMAIN}"
        seen_emails.add(email)
        # A handful are deactivated so the U20 screens have something to show;
        # one of them is the handle the scenario uses for its inactive student.
        inactive = i % 47 == 0
        accounts.append(Account(
            demo_key="cohort_inactive" if i == 0 else f"cohort_{i:03d}",
            role="student",
            email=email,
            full_name=f"{given} {surname}",
            student_id=student_id,
            password=student_id,
            status="inactive" if (inactive or i == 0) else "active",
            sort_key=100 + i,
        ))
    return accounts


def hash_all(accounts: list[Account]) -> None:
    """Argon2id is intentionally slow (~60 ms each); 200 of them serially is a
    20-second stall. argon2-cffi releases the GIL, so threads genuinely help."""
    def work(acc: Account) -> None:
        acc.password_hash = _HASHER.hash(acc.password)

    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as pool:
        list(pool.map(work, accounts))


def stage_and_seed(conn, accounts: list[Account]) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMPORARY TABLE seed_account (
                demo_key      TEXT PRIMARY KEY,
                role          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                full_name     TEXT NOT NULL,
                student_id    TEXT,
                staff_id      TEXT,
                status        TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                sort_key      INT  NOT NULL
            ) ON COMMIT DROP
        """)
        cur.executemany(
            """INSERT INTO seed_account
                   (demo_key, role, email, full_name, student_id, staff_id,
                    status, password_hash, sort_key)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(a.demo_key, a.role, a.email, a.full_name, a.student_id,
              a.staff_id, a.status, a.password_hash, a.sort_key)
             for a in accounts],
        )
        cur.execute(_SEED_SQL.read_text(encoding="utf-8"))


def report(conn) -> None:
    """Print what actually landed, so a successful run proves itself."""
    tables = [
        "user_account", "personal_info", "course", "course_enrollment",
        "attendance_session", "attendance_record", "presence_check",
        "attendance_appeal", "leave_application", "behaviour_event",
        "heatmap_snapshot", "face_embedding",
    ]
    pk = {
        "user_account": "accountid", "personal_info": "personid",
        "course": "courseid", "course_enrollment": "enrollmentid",
        "attendance_session": "attendancesessionid",
        "attendance_record": "attendancerecordid",
        "presence_check": "presencecheckid", "attendance_appeal": "appealid",
        "leave_application": "leaveapplicationid",
        "behaviour_event": "behavioureventid",
        "heatmap_snapshot": "heatmapsnapshotid", "face_embedding": "faceid",
    }
    print(f"\n{'table':24}{'rows':>7}{'id range':>16}")
    print("-" * 47)
    with conn.cursor() as cur:
        for t in tables:
            cur.execute(f"SELECT COUNT(*), MIN({pk[t]}), MAX({pk[t]}) FROM {t}")
            n, lo, hi = cur.fetchone()
            span = f"{lo}..{hi}" if n else "-"
            print(f"{t:24}{n:>7}{span:>16}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true",
                    help="actually wipe and re-seed (without it, dry run only)")
    args = ap.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set (put it in .env)", file=sys.stderr)
        return 1

    accounts = build_roster()
    students = [a for a in accounts if a.role == "student"]
    host = database_url.split("@")[-1].split("/")[0]
    print(f"target      : {host}")
    print(f"accounts    : {len(accounts)}  "
          f"({len(students)} students, {len(TEAM)} of them real team members)")
    print(f"passwords   : student ID / staff ID; admin = {ADMIN_PASSWORD!r}")

    if not args.yes:
        print("\nDry run — nothing was changed. Re-run with --yes to apply.")
        print("Sample of the roster:")
        for a in accounts[:8]:
            print(f"  {a.demo_key:16} {a.email:34} {a.full_name}")
        print(f"  … and {len(accounts) - 8} more")
        return 0

    print("\nhashing passwords (Argon2id, this takes a few seconds)…")
    hash_all(accounts)

    conn = psycopg2.connect(database_url)
    try:
        # One transaction: the staging table is ON COMMIT DROP, and a failure
        # part-way leaves the database exactly as it was.
        stage_and_seed(conn, accounts)
        conn.commit()
        report(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print("\nSeeded. Every table above should start at id 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
