-- ============================================================
-- FYP-26-S2-17 Demo seed data
-- 1 admin + 2 students + 1 course + 1 session + 2 attendance rows
-- Password for every account: demo123
-- ============================================================
-- Apply after schema.sql:
--   psql -f database/schema.sql
--   psql -f database/seed_demo.sql
-- ============================================================

-- Accounts ----------------------------------------------------
-- Argon2id hash of "demo123"  (m=64MiB, t=3, p=4)
INSERT INTO user_account (profileid, email, password_hash) VALUES
    ((SELECT profileid FROM user_profiles WHERE role='admin'   LIMIT 1),
     'admin@demo.local',   '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1),
     'alice@demo.local',   '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1),
     'bob@demo.local',     '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI')
ON CONFLICT (email) DO NOTHING;

-- Personal info -----------------------------------------------
INSERT INTO personal_info (accountid, full_name, student_id, staff_id) VALUES
    ((SELECT accountid FROM user_account WHERE email='admin@demo.local'),
     'Demo Admin',  NULL, 'A00001'),
    ((SELECT accountid FROM user_account WHERE email='alice@demo.local'),
     'Alice Tan',   'S00001', NULL),
    ((SELECT accountid FROM user_account WHERE email='bob@demo.local'),
     'Bob Lim',     'S00002', NULL)
ON CONFLICT (accountid) DO NOTHING;

-- Course + enrollment -----------------------------------------
INSERT INTO course (course_code, course_name) VALUES
    ('CSIT226', 'Final Year Project')
ON CONFLICT (course_code) DO NOTHING;

INSERT INTO course_enrollment (courseid, accountid) VALUES
    ((SELECT courseid FROM course WHERE course_code='CSIT226'),
     (SELECT accountid FROM user_account WHERE email='alice@demo.local')),
    ((SELECT courseid FROM course WHERE course_code='CSIT226'),
     (SELECT accountid FROM user_account WHERE email='bob@demo.local'))
ON CONFLICT DO NOTHING;

-- Attendance sessions -----------------------------------------
-- (a) An "ended" session yesterday so attendance history is non-empty.
INSERT INTO attendance_session (courseid, start_time, end_time, status)
SELECT courseid, NOW() - INTERVAL '1 day', NOW() - INTERVAL '23 hours', 'ended'
FROM course WHERE course_code='CSIT226'
  AND NOT EXISTS (SELECT 1 FROM attendance_session WHERE status='ended');

-- (b) An ACTIVE all-day session for today so the check-in demo works
--     any time during the day. start = today 00:00, end = tomorrow 00:00.
INSERT INTO attendance_session (courseid, start_time, end_time, status)
SELECT courseid, DATE_TRUNC('day', NOW()), DATE_TRUNC('day', NOW()) + INTERVAL '1 day', 'active'
FROM course WHERE course_code='CSIT226'
  AND NOT EXISTS (
    SELECT 1 FROM attendance_session
    WHERE status='active' AND start_time::date = CURRENT_DATE
  );

-- Attendance records ------------------------------------------
INSERT INTO attendance_record (attendancesessionid, accountid, status) VALUES
    ((SELECT attendancesessionid FROM attendance_session ORDER BY start_time DESC LIMIT 1),
     (SELECT accountid FROM user_account WHERE email='alice@demo.local'), 'present'),
    ((SELECT attendancesessionid FROM attendance_session ORDER BY start_time DESC LIMIT 1),
     (SELECT accountid FROM user_account WHERE email='bob@demo.local'),   'absent')
ON CONFLICT DO NOTHING;
