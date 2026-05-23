-- ============================================================
-- FYP-26-S2-17 Demo seed data
-- 1 admin + 2 teachers + 4 students
-- 3 courses + 3 attendance sessions + multiple attendance rows
-- 4 students enrolled across 2 courses (some in both)
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
     'admin@demo.local',     '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='teacher' LIMIT 1),
     'tara@demo.local',      '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='teacher' LIMIT 1),
     'tom@demo.local',       '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1),
     'alice@demo.local',     '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1),
     'bob@demo.local',       '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1),
     'carol@demo.local',     '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI'),
    ((SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1),
     'david@demo.local',     '$argon2id$v=19$m=65536,t=3,p=4$kcuYsgHPtrS6bcbMLnBV3g$yz7Zr4N8WiPCfId5+d+m03tiB8WYXMFj4aMPGty7ceI')
ON CONFLICT (email) DO NOTHING;

-- Personal info -----------------------------------------------
INSERT INTO personal_info (accountid, full_name, student_id, staff_id) VALUES
    ((SELECT accountid FROM user_account WHERE email='admin@demo.local'),
     'Demo Admin',   NULL,     'A00001'),
    ((SELECT accountid FROM user_account WHERE email='tara@demo.local'),
     'Tara Wong',    NULL,     'T00001'),
    ((SELECT accountid FROM user_account WHERE email='tom@demo.local'),
     'Tom Chen',     NULL,     'T00002'),
    ((SELECT accountid FROM user_account WHERE email='alice@demo.local'),
     'Alice Tan',    'S00001', NULL),
    ((SELECT accountid FROM user_account WHERE email='bob@demo.local'),
     'Bob Lim',      'S00002', NULL),
    ((SELECT accountid FROM user_account WHERE email='carol@demo.local'),
     'Carol Ng',     'S00003', NULL),
    ((SELECT accountid FROM user_account WHERE email='david@demo.local'),
     'David Koh',    'S00004', NULL)
ON CONFLICT (accountid) DO NOTHING;

-- Courses -----------------------------------------------------
INSERT INTO course (course_code, course_name) VALUES
    ('CSIT226', 'Final Year Project'),
    ('CSIT321', 'Software Engineering Project'),
    ('CSIT314', 'Software Development Methodologies')
ON CONFLICT (course_code) DO NOTHING;

-- Course enrollments ------------------------------------------
-- 4 students across 2 courses (CSIT226 + CSIT321); Alice & Carol in both.
-- CSIT314 left empty to demo an unenrolled course.
INSERT INTO course_enrollment (courseid, accountid) VALUES
    ((SELECT courseid FROM course WHERE course_code='CSIT226'),
     (SELECT accountid FROM user_account WHERE email='alice@demo.local')),
    ((SELECT courseid FROM course WHERE course_code='CSIT226'),
     (SELECT accountid FROM user_account WHERE email='bob@demo.local')),
    ((SELECT courseid FROM course WHERE course_code='CSIT226'),
     (SELECT accountid FROM user_account WHERE email='carol@demo.local')),
    ((SELECT courseid FROM course WHERE course_code='CSIT321'),
     (SELECT accountid FROM user_account WHERE email='alice@demo.local')),
    ((SELECT courseid FROM course WHERE course_code='CSIT321'),
     (SELECT accountid FROM user_account WHERE email='carol@demo.local')),
    ((SELECT courseid FROM course WHERE course_code='CSIT321'),
     (SELECT accountid FROM user_account WHERE email='david@demo.local'))
ON CONFLICT DO NOTHING;

-- Attendance sessions -----------------------------------------
-- (1) CSIT226 — ended session yesterday (history demo)
INSERT INTO attendance_session (courseid, start_time, end_time, status)
SELECT courseid, NOW() - INTERVAL '1 day', NOW() - INTERVAL '23 hours', 'ended'
FROM course WHERE course_code='CSIT226'
  AND NOT EXISTS (
    SELECT 1 FROM attendance_session s
    JOIN course c ON c.courseid = s.courseid
    WHERE c.course_code='CSIT226' AND s.status='ended'
  );

-- (2) CSIT321 — ended session 2 days ago
INSERT INTO attendance_session (courseid, start_time, end_time, status)
SELECT courseid, NOW() - INTERVAL '2 days', NOW() - INTERVAL '2 days' + INTERVAL '1 hour', 'ended'
FROM course WHERE course_code='CSIT321'
  AND NOT EXISTS (
    SELECT 1 FROM attendance_session s
    JOIN course c ON c.courseid = s.courseid
    WHERE c.course_code='CSIT321' AND s.status='ended'
  );

-- (3) CSIT226 — ACTIVE all-day session today (check-in demo)
INSERT INTO attendance_session (courseid, start_time, end_time, status)
SELECT courseid, DATE_TRUNC('day', NOW()), DATE_TRUNC('day', NOW()) + INTERVAL '1 day', 'active'
FROM course WHERE course_code='CSIT226'
  AND NOT EXISTS (
    SELECT 1 FROM attendance_session s
    JOIN course c ON c.courseid = s.courseid
    WHERE c.course_code='CSIT226' AND s.status='active'
      AND s.start_time::date = CURRENT_DATE
  );

-- Attendance records ------------------------------------------
-- CSIT226 ended session (yesterday): all 3 enrolled students
INSERT INTO attendance_record (attendancesessionid, accountid, status)
SELECT s.attendancesessionid, ua.accountid, v.status
FROM attendance_session s
JOIN course c ON c.courseid = s.courseid AND c.course_code='CSIT226' AND s.status='ended'
CROSS JOIN (VALUES
    ('alice@demo.local', 'present'),
    ('bob@demo.local',   'late'),
    ('carol@demo.local', 'absent')
) AS v(email, status)
JOIN user_account ua ON ua.email = v.email
ON CONFLICT DO NOTHING;

-- CSIT321 ended session (2 days ago): all 3 enrolled students
INSERT INTO attendance_record (attendancesessionid, accountid, status)
SELECT s.attendancesessionid, ua.accountid, v.status
FROM attendance_session s
JOIN course c ON c.courseid = s.courseid AND c.course_code='CSIT321' AND s.status='ended'
CROSS JOIN (VALUES
    ('alice@demo.local', 'present'),
    ('carol@demo.local', 'present'),
    ('david@demo.local', 'absent')
) AS v(email, status)
JOIN user_account ua ON ua.email = v.email
ON CONFLICT DO NOTHING;

-- CSIT226 active session (today): partial check-ins so far
INSERT INTO attendance_record (attendancesessionid, accountid, status)
SELECT s.attendancesessionid, ua.accountid, v.status
FROM attendance_session s
JOIN course c ON c.courseid = s.courseid AND c.course_code='CSIT226' AND s.status='active'
CROSS JOIN (VALUES
    ('alice@demo.local', 'present'),
    ('bob@demo.local',   'present')
) AS v(email, status)
JOIN user_account ua ON ua.email = v.email
ON CONFLICT DO NOTHING;
