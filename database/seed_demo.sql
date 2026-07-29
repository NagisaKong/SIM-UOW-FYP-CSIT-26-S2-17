-- ============================================================
-- FYP-26-S2-17 Demo seed data  (aligned with the live demo script)
-- ============================================================
-- Goal: keep the key accounts the demo script relies on
--   (admin@demo.local / tara@demo.local / alice@demo.local /
--    bob@demo.local, all sharing the password whose Argon2id hash is
--    injected via the `demo_password_hash` psql variable) while populating
--   EVERY data state the system can ever produce, so every screen
--   (Users, Courses, Sessions, Live, Roster, Analytics, Reports,
--    Appeals, Leave, Face DB, Behaviour) has realistic content.
--
-- Coverage matrix
--   user_account.status ......... active + inactive           (U20)
--   user_profiles roles ......... student / teacher / admin
--   course.status ............... active + inactive
--   course.teacher_id ........... assigned + unassigned       (U26)
--   course_enrollment.status .... active / withdrawn / completed
--   attendance_session.status ... scheduled / active / ended / cancelled
--   attendance_record.status .... present / late / absent / leave / early_left
--   attendance_appeal.status .... pending / approved / rejected   (U08)
--   leave_application.status .... pending / approved / rejected   (U28/U31)
--   presence_check .............. multi-window scan (present/late/early_left)
--   behaviour_event.event_type .. drowsiness / phone / distraction / other (U32)
--   heatmap_snapshot ............ spatial grid                  (U33)
--   behaviour_config ............ enabled + disabled course     (U35)
--   model_configs ............... active + inactive model       (U34)
--   face_embedding .............. active + inactive, synthetic   (U19/U21)
--   session_recording ........... 30-day retained recording     (U03)
--
-- NOTE: course code DEMO101 is intentionally LEFT FREE — the demo
--       script creates it live in Act 1 (admin builds the course).
--
-- This script is fully idempotent: it deletes prior demo rows in
-- FK-safe order, then re-inserts deterministic data. Safe to re-run.
--
-- Apply after schema.sql. The demo-account password hash is NOT hardcoded:
-- it must be injected at apply time via the `demo_password_hash` psql
-- variable, so every deployment chooses its own password.
--
--   # 1. generate an Argon2id hash for your chosen demo password:
--   HASH=$(python -c "from argon2 import PasswordHasher, Type; \
--     print(PasswordHasher(type=Type.ID, memory_cost=65536, time_cost=3, \
--     parallelism=4).hash('YOUR_DEMO_PASSWORD'))")
--
--   # 2. apply the seed:
--   psql -f database/schema.sql
--   psql -v demo_password_hash="$HASH" -f database/seed_demo.sql
-- ============================================================

-- Fail loudly if the password hash variable was not provided.
\if :{?demo_password_hash}
\else
  \echo '>>> ERROR: psql variable `demo_password_hash` is not set.'
  \echo '>>> Generate an Argon2id hash and re-run with:'
  \echo '>>>   psql -v demo_password_hash="<argon2id-hash>" -f database/seed_demo.sql'
  \quit
\endif

-- Stash the injected hash in a session GUC so the DO block can read it
-- (psql does not interpolate :variables inside dollar-quoted blocks).
SELECT set_config('app.demo_password_hash', :'demo_password_hash', false);

DO $seed$
DECLARE
    -- Argon2id password hash for all demo accounts, injected at apply time.
    v_pwd  TEXT := current_setting('app.demo_password_hash');

    p_student INT;
    p_teacher INT;
    p_admin   INT;

    -- accounts
    a_admin INT; a_tara INT; a_tom INT; a_ian INT;
    a_alice INT; a_bob INT; a_carol INT; a_david INT; a_eve INT; a_frank INT;

    -- courses
    c226 INT; c321 INT; c314 INT; c113 INT;

    -- sessions
    s226_active INT; s226_yest INT; s226_old INT; s226_future INT; s226_cancel INT;
    s321_2d INT; s321_future INT; s314_future INT;

    -- attendance records we later attach appeals to
    ar_bob_absent INT; ar_carol_late INT; ar_eve_absent INT;

    -- "midnight today" anchor → deterministic, re-runnable timestamps
    t0 TIMESTAMPTZ := DATE_TRUNC('day', NOW());

    rec RECORD;
BEGIN
    -- ========================================================
    -- 0. Clean prior demo data (child → parent, FK-safe)
    -- ========================================================
    DELETE FROM behaviour_event;
    DELETE FROM heatmap_snapshot;
    DELETE FROM presence_check;
    DELETE FROM session_recording;
    DELETE FROM leave_application;
    DELETE FROM attendance_appeal;
    DELETE FROM attendance_record;
    DELETE FROM attendance_session;
    DELETE FROM behaviour_config;
    DELETE FROM course_enrollment;
    DELETE FROM model_configs;
    UPDATE attendance_threshold_config SET updated_by = NULL;
    DELETE FROM course;          -- teacher_id FK cleared by deleting courses first
    DELETE FROM face_embedding;
    DELETE FROM personal_info;   -- cascades from user_account too, explicit for clarity
    DELETE FROM user_account;

    -- Restart the identity counters of every table we just emptied, so a
    -- re-seeded demo database numbers rows from 1 again. DELETE (unlike
    -- TRUNCATE ... RESTART IDENTITY, which FK dependencies make awkward
    -- here) leaves sequences untouched, so without this the FaceIDs,
    -- AccountIDs, … keep climbing across every reset and demo screenshots
    -- show arbitrary numbers. USER_PROFILES is deliberately excluded — it is
    -- seeded by schema.sql and never deleted above.
    FOR rec IN
        SELECT s.relname AS seqname
        FROM pg_class s
        JOIN pg_depend d ON d.objid = s.oid AND d.deptype = 'a'
        JOIN pg_class t ON t.oid = d.refobjid
        WHERE s.relkind = 'S'
          AND t.relname = ANY (ARRAY[
              'behaviour_event', 'heatmap_snapshot', 'presence_check',
              'session_recording', 'leave_application', 'attendance_appeal',
              'attendance_record', 'attendance_session', 'course_enrollment',
              'model_configs', 'course', 'face_embedding', 'personal_info',
              'user_account'
          ])
    LOOP
        EXECUTE format('ALTER SEQUENCE %I RESTART WITH 1', rec.seqname);
    END LOOP;

    -- role profile ids (seeded by schema.sql)
    SELECT profileid INTO p_student FROM user_profiles WHERE role='student' LIMIT 1;
    SELECT profileid INTO p_teacher FROM user_profiles WHERE role='teacher' LIMIT 1;
    SELECT profileid INTO p_admin   FROM user_profiles WHERE role='admin'   LIMIT 1;

    -- ========================================================
    -- 1. Accounts  (active + inactive — U20)
    -- ========================================================
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_admin, 'admin@demo.local', v_pwd, 'active')   RETURNING accountid INTO a_admin;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_teacher, 'tara@demo.local', v_pwd, 'active')  RETURNING accountid INTO a_tara;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_teacher, 'tom@demo.local', v_pwd, 'active')   RETURNING accountid INTO a_tom;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_teacher, 'ian@demo.local', v_pwd, 'inactive') RETURNING accountid INTO a_ian;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_student, 'zyu010@mymail.sim.edu.sg', v_pwd, 'active') RETURNING accountid INTO a_alice;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_student, 'jzhang092@mymail.sim.edu.sg', v_pwd, 'active') RETURNING accountid INTO a_bob;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_student, 'carol@demo.local', v_pwd, 'active') RETURNING accountid INTO a_carol;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_student, 'david@demo.local', v_pwd, 'active') RETURNING accountid INTO a_david;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_student, 'eve@demo.local', v_pwd, 'active')   RETURNING accountid INTO a_eve;
    INSERT INTO user_account (profileid, email, password_hash, status)
        VALUES (p_student, 'frank@demo.local', v_pwd, 'inactive') RETURNING accountid INTO a_frank;

    -- ========================================================
    -- 2. Personal info
    -- ========================================================
    INSERT INTO personal_info (accountid, full_name, student_id, staff_id) VALUES
        (a_admin, 'Demo Admin', NULL,     'A00001'),
        (a_tara,  'Tara Wong',  NULL,     'T00001'),
        (a_tom,   'Tom Chen',   NULL,     'T00002'),
        (a_ian,   'Ian Lee',    NULL,     'T00003'),
        (a_alice, 'yu zhanghao',  'S00001', NULL),
        (a_bob,   'zhang jiqian', 'S00002', NULL),
        (a_carol, 'Carol Ng',   'S00003', NULL),
        (a_david, 'David Koh',  'S00004', NULL),
        (a_eve,   'Eve Sim',    'S00005', NULL),
        (a_frank, 'Frank Ong',  'S00006', NULL);

    -- ========================================================
    -- 3. Courses  (active/inactive, assigned/unassigned teacher — U26)
    -- ========================================================
    INSERT INTO course (course_code, course_name, status, teacher_id)
        VALUES ('CSIT226', 'Final Year Project',                  'active',   a_tara) RETURNING courseid INTO c226;
    INSERT INTO course (course_code, course_name, status, teacher_id)
        VALUES ('CSIT321', 'Software Engineering Project',        'active',   a_tara) RETURNING courseid INTO c321;
    INSERT INTO course (course_code, course_name, status, teacher_id)
        VALUES ('CSIT314', 'Software Development Methodologies',  'active',   a_tom)  RETURNING courseid INTO c314;
    INSERT INTO course (course_code, course_name, status, teacher_id)
        VALUES ('CSIT113', 'Problem Solving (archived)',          'inactive', NULL)   RETURNING courseid INTO c113;
    -- DEMO101 deliberately omitted → created live during the demo (Act 1).
    -- CSIT314 has no enrollments → demonstrates an empty course.

    -- ========================================================
    -- 4. Enrollments  (active / withdrawn / completed)
    -- ========================================================
    INSERT INTO course_enrollment (courseid, accountid, status) VALUES
        (c226, a_alice, 'active'),
        (c226, a_bob,   'active'),
        (c226, a_carol, 'active'),
        (c226, a_eve,   'active'),
        (c226, a_david, 'withdrawn'),     -- dropped CSIT226
        (c321, a_alice, 'active'),
        (c321, a_carol, 'active'),
        (c321, a_david, 'active'),
        (c321, a_bob,   'completed');     -- finished CSIT321

    -- ========================================================
    -- 5. Attendance sessions  (every status)
    -- ========================================================
    -- CSIT226 — ACTIVE today (the Live-scan demo target / fallback)
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c226, t0 + INTERVAL '9 hours', NULL, 'active')          RETURNING attendancesessionid INTO s226_active;
    -- CSIT226 — ENDED yesterday
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c226, t0 - INTERVAL '1 day' + INTERVAL '9 hours',
                       t0 - INTERVAL '1 day' + INTERVAL '11 hours', 'ended') RETURNING attendancesessionid INTO s226_yest;
    -- CSIT226 — ENDED 8 days ago
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c226, t0 - INTERVAL '8 days' + INTERVAL '9 hours',
                       t0 - INTERVAL '8 days' + INTERVAL '11 hours', 'ended') RETURNING attendancesessionid INTO s226_old;
    -- CSIT226 — SCHEDULED tomorrow
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c226, t0 + INTERVAL '1 day' + INTERVAL '9 hours',
                       t0 + INTERVAL '1 day' + INTERVAL '11 hours', 'scheduled') RETURNING attendancesessionid INTO s226_future;
    -- CSIT226 — CANCELLED (3 days ago)
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c226, t0 - INTERVAL '3 days' + INTERVAL '9 hours',
                       t0 - INTERVAL '3 days' + INTERVAL '11 hours', 'cancelled') RETURNING attendancesessionid INTO s226_cancel;
    -- CSIT321 — ENDED 2 days ago
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c321, t0 - INTERVAL '2 days' + INTERVAL '14 hours',
                       t0 - INTERVAL '2 days' + INTERVAL '15 hours', 'ended') RETURNING attendancesessionid INTO s321_2d;
    -- CSIT321 — SCHEDULED in 2 days
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c321, t0 + INTERVAL '2 days' + INTERVAL '14 hours',
                       t0 + INTERVAL '2 days' + INTERVAL '15 hours', 'scheduled') RETURNING attendancesessionid INTO s321_future;
    -- CSIT314 — SCHEDULED (empty course)
    INSERT INTO attendance_session (courseid, start_time, end_time, status)
        VALUES (c314, t0 + INTERVAL '3 days' + INTERVAL '10 hours',
                       t0 + INTERVAL '3 days' + INTERVAL '11 hours', 'scheduled') RETURNING attendancesessionid INTO s314_future;

    -- ========================================================
    -- 6. Attendance records  (present / late / absent / leave / early_left)
    -- ========================================================
    -- CSIT226 ended yesterday
    INSERT INTO attendance_record (attendancesessionid, accountid, status, marked_at)
        VALUES (s226_yest, a_alice, 'present', t0 - INTERVAL '1 day' + INTERVAL '9 hours 3 minutes');
    INSERT INTO attendance_record (attendancesessionid, accountid, status, marked_at)
        VALUES (s226_yest, a_bob, 'absent', t0 - INTERVAL '1 day' + INTERVAL '11 hours')
        RETURNING attendancerecordid INTO ar_bob_absent;
    INSERT INTO attendance_record (attendancesessionid, accountid, status, marked_at)
        VALUES (s226_yest, a_carol, 'late', t0 - INTERVAL '1 day' + INTERVAL '9 hours 25 minutes')
        RETURNING attendancerecordid INTO ar_carol_late;
    INSERT INTO attendance_record (attendancesessionid, accountid, status, marked_at)
        VALUES (s226_yest, a_eve, 'leave', t0 - INTERVAL '1 day' + INTERVAL '9 hours');

    -- CSIT226 ended 8 days ago
    INSERT INTO attendance_record (attendancesessionid, accountid, status) VALUES
        (s226_old, a_alice, 'present'),
        (s226_old, a_bob,   'present'),
        (s226_old, a_carol, 'early_left');
    INSERT INTO attendance_record (attendancesessionid, accountid, status)
        VALUES (s226_old, a_eve, 'absent')
        RETURNING attendancerecordid INTO ar_eve_absent;

    -- CSIT321 ended 2 days ago
    INSERT INTO attendance_record (attendancesessionid, accountid, status) VALUES
        (s321_2d, a_alice, 'present'),
        (s321_2d, a_carol, 'present'),
        (s321_2d, a_david, 'absent');

    -- CSIT226 active today — partial check-ins "so far"
    INSERT INTO attendance_record (attendancesessionid, accountid, status) VALUES
        (s226_active, a_alice, 'present'),
        (s226_active, a_bob,   'present');

    -- ========================================================
    -- 7. Appeals  (pending / approved / rejected — U08)
    -- ========================================================
    INSERT INTO attendance_appeal (attendancerecordid, accountid, reason, status)
        VALUES (ar_bob_absent, a_bob,
                'I was present but the camera did not detect me near the back row.', 'pending');
    INSERT INTO attendance_appeal (attendancerecordid, accountid, reason, status, reviewed_by, reviewed_at)
        VALUES (ar_carol_late, a_carol,
                'Public transport delay — boarded the next bus immediately.', 'approved',
                a_tara, NOW() - INTERVAL '20 hours');
    INSERT INTO attendance_appeal (attendancerecordid, accountid, reason, status, reviewed_by, reviewed_at)
        VALUES (ar_eve_absent, a_eve,
                'Thought the lecture was online.', 'rejected',
                a_tara, NOW() - INTERVAL '6 days');

    -- ========================================================
    -- 8. Leave applications  (pending / approved / rejected — U28/U31)
    -- ========================================================
    -- Eve's approved leave underpins her 'leave' attendance record above.
    INSERT INTO leave_application (accountid, attendancesessionid, reason, status, reviewed_by, reviewed_at, reviewer_comment)
        VALUES (a_eve, s226_yest, 'Medical appointment with supporting MC.', 'approved',
                a_tara, NOW() - INTERVAL '1 day' - INTERVAL '2 hours', 'MC verified — approved.');
    INSERT INTO leave_application (accountid, attendancesessionid, reason, status)
        VALUES (a_alice, s226_future, 'Family event overseas; will submit documents.', 'pending');
    INSERT INTO leave_application (accountid, attendancesessionid, reason, status, reviewed_by, reviewed_at, reviewer_comment)
        VALUES (a_david, s321_future, 'Personal reasons.', 'rejected',
                a_tara, NOW() - INTERVAL '1 hour', 'Insufficient justification.');

    -- ========================================================
    -- 9. Presence-check snapshots for the ACTIVE CSIT226 session (U03/U16)
    -- Three windows (~35 / 20 / 5 min ago). On "End Scan & Finalize":
    --     Alice → present     (seen every window)
    --     Bob   → late        (missing from first window)
    --     Carol → early_left  (missing from last window)
    -- ========================================================
    INSERT INTO presence_check (attendancesessionid, accountid, detected, detected_at, confidence, camera_id)
    SELECT s226_active, v.acc, v.detected,
           NOW() - v.mins * INTERVAL '1 minute',
           CASE WHEN v.detected THEN 0.82 ELSE NULL END,
           'demo-cam-1'
    FROM (VALUES
        (a_alice, 35, TRUE),  (a_alice, 20, TRUE),  (a_alice, 5, TRUE),
        (a_bob,   35, FALSE), (a_bob,   20, TRUE),  (a_bob,   5, TRUE),
        (a_carol, 35, TRUE),  (a_carol, 20, TRUE),  (a_carol, 5, FALSE)
    ) AS v(acc, mins, detected);

    -- ========================================================
    -- 10. Behaviour events for the ACTIVE session (every type — U32)
    -- ========================================================
    INSERT INTO behaviour_event (attendancesessionid, accountid, event_type, detected_at, duration_seconds, confidence, metadata) VALUES
        (s226_active, a_alice, 'drowsiness',  NOW() - INTERVAL '18 minutes', 12, 0.78, '{"eye_aspect_ratio": 0.18}'::jsonb),
        (s226_active, a_bob,   'phone',       NOW() - INTERVAL '12 minutes', 30, 0.83, '{"object": "cellphone"}'::jsonb),
        (s226_active, a_carol, 'distraction', NOW() - INTERVAL '9 minutes',   8, 0.66, '{"gaze": "off-screen"}'::jsonb),
        (s226_active, a_eve,   'other',       NOW() - INTERVAL '4 minutes',   5, 0.55, '{"note": "talking to neighbour"}'::jsonb);

    -- ========================================================
    -- 11. Heatmap snapshot — 3x3 activity grid (U33)
    -- ========================================================
    INSERT INTO heatmap_snapshot (attendancesessionid, zone_x, zone_y, intensity, captured_at, camera_id)
    SELECT s226_active, gx, gy,
           ROUND(((gx + gy + 1)::numeric / 6.0), 3)::float,
           NOW() - INTERVAL '5 minutes', 'demo-cam-1'
    FROM generate_series(0, 2) gx CROSS JOIN generate_series(0, 2) gy;

    -- ========================================================
    -- 12. Behaviour config — enabled vs disabled course (U35)
    -- ========================================================
    INSERT INTO behaviour_config (courseid, enabled, drowsiness, phone_usage, heatmap, updated_by) VALUES
        (c226, TRUE,  TRUE,  TRUE,  TRUE,  a_tara),
        (c321, FALSE, TRUE,  FALSE, FALSE, a_tara);
    -- CSIT314 left at table defaults (no row) → analysis off.

    -- ========================================================
    -- 13. Model configs — active + inactive (U34)
    -- ========================================================
    INSERT INTO model_configs (model_name, similarity_threshold, is_active, updated_by) VALUES
        ('arcface_ensemble', 0.35, TRUE,  a_admin),
        ('facenet',          0.40, FALSE, a_admin);

    -- Institution-wide threshold config (single row id=1 from schema).
    UPDATE attendance_threshold_config
       SET minimum_attendance_rate    = 70.0,
           absence_threshold          = 3,
           late_grace_seconds         = 600,
           detection_interval_seconds = 1200,
           updated_by                 = a_admin
     WHERE configid = 1;

    -- ========================================================
    -- 14. Face embeddings — synthetic, active + inactive (U19/U21)
    -- ========================================================
    -- Alice & Bob are intentionally NOT pre-enrolled: the presenter
    -- enrols their faces LIVE in Act 1 (U19) and the live scan then
    -- recognises them. Carol/David/Eve/Frank populate the Face DB tab.
    -- Vectors are synthetic placeholders (is_synthetic = TRUE): each component
    -- is ((g + seed) % 13) * 0.07, i.e. non-negative, so any two of them are
    -- 0.58–0.78 cosine-similar to each other. They exist only to give the Face
    -- DB screen realistic rows — EmbeddingRepo.load_active_embeddings excludes
    -- is_synthetic rows from live matching (AI_MATCH_SYNTHETIC), so they can
    -- never be mistaken for a real student during a scan.
    FOR rec IN
        SELECT * FROM (VALUES
            (a_carol, 3,  TRUE),
            (a_david, 5,  TRUE),
            (a_eve,   7,  TRUE),
            (a_frank, 9,  TRUE),
            (a_david, 11, FALSE)    -- David: an older, deactivated embedding
        ) AS v(acc, seed, act)
    LOOP
        INSERT INTO face_embedding
            (accountid, embedding_vector, model_name, model_version, dimension,
             is_active, is_synthetic, consent_given_at, retention_until)
        VALUES (
            rec.acc,
            ('[' || (SELECT string_agg(((((g + rec.seed) % 13) * 0.07))::numeric(6,4)::text, ',')
                       FROM generate_series(1, 512) g) || ']')::vector,
            'arcface', 'r100', 512,
            rec.act, TRUE, NOW(), (NOW() + INTERVAL '365 days')::date
        );
    END LOOP;

    -- ========================================================
    -- 15. Session recording for the ACTIVE session (30-day retained — U03)
    -- ========================================================
    INSERT INTO session_recording (attendancesessionid, file_path)
        VALUES (s226_active, 'recordings/session_' || s226_active || '.mp4');

END
$seed$;
