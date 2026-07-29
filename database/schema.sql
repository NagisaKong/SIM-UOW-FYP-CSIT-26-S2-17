-- ============================================================
-- FYP-26-S2-17: Face Recognition Attendance System
-- Database Schema (PostgreSQL)
-- Vision 0.5
-- ============================================================
-- Extension: pgvector for efficient facial embedding storage/search
-- Run once: CREATE EXTENSION IF NOT EXISTS vector;
-- ============================================================

-- ------------------------------------------------------------
-- 1. USER_PROFILES  (role definitions – seed before accounts)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS USER_PROFILES (
    ProfileID   SERIAL          PRIMARY KEY,
    Role        VARCHAR(20)     NOT NULL CHECK (Role IN ('student', 'teacher', 'admin')),
    Description TEXT,
    status      VARCHAR(20)     NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'inactive'))
);

-- ------------------------------------------------------------
-- 2. USER_ACCOUNT  (authentication credentials)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS USER_ACCOUNT (
    AccountID     SERIAL          PRIMARY KEY,
    ProfileID     INTEGER         NOT NULL REFERENCES USER_PROFILES(ProfileID),
    email         VARCHAR(255)    NOT NULL UNIQUE,
    password_hash VARCHAR(255)    NOT NULL,
    -- U20: per-account activate/deactivate switch (USER_PROFILES.status is
    -- role-wide because all accounts of a role share one profile row).
    status        VARCHAR(20)     NOT NULL DEFAULT 'active'
                                  CHECK (status IN ('active', 'inactive')),
    created_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE USER_ACCOUNT ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active';

CREATE INDEX IF NOT EXISTS idx_user_account_profile ON USER_ACCOUNT(ProfileID);

-- ------------------------------------------------------------
-- 3. PERSONAL_INFO
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS PERSONAL_INFO (
    PersonID    SERIAL          PRIMARY KEY,
    AccountID   INTEGER         NOT NULL UNIQUE REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    full_name   VARCHAR(255)    NOT NULL,
    student_id  VARCHAR(50),    -- NULL for teachers/admins
    staff_id    VARCHAR(50),    -- NULL for students
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Enforce that at least one ID type is present
    CONSTRAINT chk_personal_id CHECK ((student_id IS NOT NULL AND staff_id IS NULL) OR (student_id IS NULL AND staff_id IS NOT NULL))
);

-- ------------------------------------------------------------
-- 4. FACE_EMBEDDING  (biometric data)
-- ------------------------------------------------------------
-- embedding_vector is a pgvector VECTOR(512). BOTH recognition models used
-- here emit 512-d embeddings, so one column width serves both:
--   arcface — InsightFace buffalo_l / r100                        → 512
--   facenet — facenet-pytorch InceptionResnetV1(pretrained=
--             'vggface2'); its final layer is Linear(1792, 512)   → 512
-- (The original 2015 FaceNet paper used 128-d; the PyTorch port does not.
--  Verify with: InceptionResnetV1().last_linear.out_features)
-- One row per model per account — model_name differentiates them, and
-- EmbeddingRepo.load_active_embeddings skips any row whose stored vector
-- length disagrees with its `dimension` column.
CREATE TABLE IF NOT EXISTS FACE_EMBEDDING (
    FaceID           SERIAL          PRIMARY KEY,
    AccountID        INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    embedding_vector VECTOR(512)     NOT NULL,   -- replace with vector(N) after pgvector install
    model_name       VARCHAR(100)    NOT NULL,   -- e.g. 'arcface', 'facenet'
    model_version    VARCHAR(50)     NOT NULL,   -- e.g. 'r100', '20180402-114759'
    dimension        INTEGER         NOT NULL,   -- 512 for ArcFace, 128 for FaceNet
    is_active        BOOLEAN         NOT NULL DEFAULT TRUE,
    is_synthetic     BOOLEAN         NOT NULL DEFAULT FALSE,
    -- PDPC biometric consent & retention fields
    consent_given_at TIMESTAMPTZ,
    retention_until  DATE,
    created_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_embedding_account ON FACE_EMBEDDING(AccountID);
CREATE INDEX IF NOT EXISTS idx_face_embedding_active  ON FACE_EMBEDDING(AccountID, is_active);

-- ------------------------------------------------------------
-- 5. COURSE
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS COURSE (
    CourseID    SERIAL          PRIMARY KEY,
    course_code VARCHAR(20)     NOT NULL UNIQUE,
    course_name VARCHAR(255)    NOT NULL,
    status      VARCHAR(20)     NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'inactive')),
    teacher_id  INTEGER         REFERENCES USER_ACCOUNT(AccountID)  -- assigned lecturer (U26)
);

ALTER TABLE COURSE ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active';
ALTER TABLE COURSE ADD COLUMN IF NOT EXISTS teacher_id INTEGER REFERENCES USER_ACCOUNT(AccountID);

-- ------------------------------------------------------------
-- 6. COURSE_ENROLLMENT  (student ↔ course membership)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS COURSE_ENROLLMENT (
    EnrollmentID SERIAL          PRIMARY KEY,
    CourseID     INTEGER         NOT NULL REFERENCES COURSE(CourseID) ON DELETE CASCADE,
    AccountID    INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    Status       VARCHAR(20)     NOT NULL DEFAULT 'active'
                                 CHECK (Status IN ('active', 'withdrawn', 'completed')),
    UNIQUE (CourseID, AccountID)
);

CREATE INDEX IF NOT EXISTS idx_enrollment_course  ON COURSE_ENROLLMENT(CourseID);
CREATE INDEX IF NOT EXISTS idx_enrollment_account ON COURSE_ENROLLMENT(AccountID);

-- ------------------------------------------------------------
-- 7. ATTENDANCE_SESSION  (a single lecture / tutorial slot)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ATTENDANCE_SESSION (
    AttendanceSessionID SERIAL          PRIMARY KEY,
    CourseID            INTEGER         NOT NULL REFERENCES COURSE(CourseID),
    start_time          TIMESTAMPTZ     NOT NULL,
    end_time            TIMESTAMPTZ,
    status              VARCHAR(20)     NOT NULL DEFAULT 'scheduled'
                                        CHECK (status IN ('scheduled', 'active', 'ended', 'cancelled')),
    CONSTRAINT chk_session_times CHECK (end_time IS NULL OR end_time > start_time)
);

CREATE INDEX IF NOT EXISTS idx_session_course  ON ATTENDANCE_SESSION(CourseID);
CREATE INDEX IF NOT EXISTS idx_session_status  ON ATTENDANCE_SESSION(status);

-- ------------------------------------------------------------
-- 8. ATTENDANCE_RECORD  (one row per student per session)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ATTENDANCE_RECORD (
    AttendanceRecordID  SERIAL          PRIMARY KEY,
    AttendanceSessionID INTEGER         NOT NULL REFERENCES ATTENDANCE_SESSION(AttendanceSessionID),
    AccountID           INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID),
    marked_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    status              VARCHAR(20)     NOT NULL
                                        CHECK (status IN ('present', 'late', 'absent')),
    UNIQUE (AttendanceSessionID, AccountID)
);

CREATE INDEX IF NOT EXISTS idx_record_session ON ATTENDANCE_RECORD(AttendanceSessionID);
CREATE INDEX IF NOT EXISTS idx_record_account ON ATTENDANCE_RECORD(AccountID);

-- ------------------------------------------------------------
-- 9. ATTENDANCE_APPEAL  (student appeals – user story requirement)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ATTENDANCE_APPEAL (
    AppealID            SERIAL          PRIMARY KEY,
    AttendanceRecordID  INTEGER         NOT NULL REFERENCES ATTENDANCE_RECORD(AttendanceRecordID),
    AccountID           INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID),
    reason              TEXT            NOT NULL,
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending'
                                        CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by         INTEGER         REFERENCES USER_ACCOUNT(AccountID),  -- teacher/admin
    reviewed_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CHECK ((status = 'pending' AND reviewed_by IS NULL AND reviewed_at IS NULL) OR (status IN ('approved', 'rejected') AND reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_appeal_record  ON ATTENDANCE_APPEAL(AttendanceRecordID);
CREATE INDEX IF NOT EXISTS idx_appeal_account ON ATTENDANCE_APPEAL(AccountID);
CREATE INDEX IF NOT EXISTS idx_appeal_status  ON ATTENDANCE_APPEAL(status);

-- ------------------------------------------------------------
-- 10. MODEL_CONFIGS (Stores fine-tuned AI thresholds)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS MODEL_CONFIGS (
    ConfigID          SERIAL          PRIMARY KEY,
    model_name        VARCHAR(100)    NOT NULL, -- e.g., 'arcface_ensemble'
    similarity_threshold FLOAT        NOT NULL DEFAULT 0.35, -- The tweaked value
    is_active         BOOLEAN         NOT NULL DEFAULT TRUE,
    updated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_by        INTEGER         REFERENCES USER_ACCOUNT(AccountID)
);

-- The updated_at trigger for MODEL_CONFIGS is installed by the
-- DO block further below (model_configs is in its target array).

-- ============================================================
-- Seed Data: default role profiles
-- ============================================================
INSERT INTO USER_PROFILES (Role, Description, status) VALUES
    ('student', 'Enrolled student — attendance subject',  'active'),
    ('teacher', 'UOW lecturer — manages sessions',        'active'),
    ('admin',   'System administrator — full access',     'active')
ON CONFLICT DO NOTHING;

-- ============================================================
-- updated_at auto-maintenance trigger
-- ============================================================
CREATE OR REPLACE FUNCTION trg_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'user_account', 'personal_info', 'face_embedding', 'attendance_appeal', 'model_configs'
    ] LOOP
        EXECUTE format('
            CREATE OR REPLACE TRIGGER trg_%s_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();
        ', t, t);
    END LOOP;
END;
$$;

-- ============================================================
-- v0.5 additions — extra tables/columns for U16/U28/U31/U32/U33/U34/U35
-- All blocks are idempotent (IF NOT EXISTS / DROP+ADD constraint).
-- ============================================================

-- ------------------------------------------------------------
-- 11. LEAVE_APPLICATION  (U28 student submit, U31 teacher review)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS LEAVE_APPLICATION (
    LeaveApplicationID  SERIAL          PRIMARY KEY,
    AccountID           INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    AttendanceSessionID INTEGER         NOT NULL REFERENCES ATTENDANCE_SESSION(AttendanceSessionID) ON DELETE CASCADE,
    reason              TEXT            NOT NULL,
    supporting_doc_url  TEXT,
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending'
                                        CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by         INTEGER         REFERENCES USER_ACCOUNT(AccountID),
    reviewed_at         TIMESTAMPTZ,
    reviewer_comment    TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (AccountID, AttendanceSessionID),
    CHECK ((status = 'pending'  AND reviewed_by IS NULL     AND reviewed_at IS NULL)
        OR (status IN ('approved', 'rejected') AND reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_leave_session  ON LEAVE_APPLICATION(AttendanceSessionID);
CREATE INDEX IF NOT EXISTS idx_leave_account  ON LEAVE_APPLICATION(AccountID);
CREATE INDEX IF NOT EXISTS idx_leave_status   ON LEAVE_APPLICATION(status);


-- ------------------------------------------------------------
-- 12. PRESENCE_CHECK  (U16 periodic in-class scans for early-left)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS PRESENCE_CHECK (
    PresenceCheckID     SERIAL          PRIMARY KEY,
    AttendanceSessionID INTEGER         NOT NULL REFERENCES ATTENDANCE_SESSION(AttendanceSessionID) ON DELETE CASCADE,
    AccountID           INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    detected            BOOLEAN         NOT NULL,           -- TRUE = seen on this scan
    detected_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    camera_id           VARCHAR(50),                        -- which internal camera fired the scan
    confidence          FLOAT
);

CREATE INDEX IF NOT EXISTS idx_presence_session_acc
    ON PRESENCE_CHECK(AttendanceSessionID, AccountID);
CREATE INDEX IF NOT EXISTS idx_presence_session_time
    ON PRESENCE_CHECK(AttendanceSessionID, detected_at);


-- ------------------------------------------------------------
-- 13. BEHAVIOUR_EVENT  (U32 drowsiness / phone-use events)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS BEHAVIOUR_EVENT (
    BehaviourEventID    SERIAL          PRIMARY KEY,
    AttendanceSessionID INTEGER         NOT NULL REFERENCES ATTENDANCE_SESSION(AttendanceSessionID) ON DELETE CASCADE,
    AccountID           INTEGER         NOT NULL REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    event_type          VARCHAR(30)     NOT NULL
                                        CHECK (event_type IN ('drowsiness', 'phone', 'distraction', 'other')),
    detected_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    duration_seconds    INTEGER,
    confidence          FLOAT,
    metadata            JSONB
);

CREATE INDEX IF NOT EXISTS idx_behaviour_session
    ON BEHAVIOUR_EVENT(AttendanceSessionID);
CREATE INDEX IF NOT EXISTS idx_behaviour_session_acc
    ON BEHAVIOUR_EVENT(AttendanceSessionID, AccountID);
CREATE INDEX IF NOT EXISTS idx_behaviour_type
    ON BEHAVIOUR_EVENT(event_type);


-- ------------------------------------------------------------
-- 13b. BEHAVIOUR_COVERAGE  (U32 analysis coverage per student)
-- ------------------------------------------------------------
-- Behaviour events only record what WAS detected. To distinguish "no
-- drowsiness/phone observed" from "this student could never be analysed"
-- (face too small / occluded / outside the internal camera's view), the
-- service also counts, per student per session, how many ~1 fps samples
-- recognised them (samples_total) and how many of those yielded usable
-- facial landmarks (samples_analysed). U32 reports a student whose
-- samples_analysed is 0 as *inconclusive* rather than as well-behaved.
CREATE TABLE IF NOT EXISTS BEHAVIOUR_COVERAGE (
    AttendanceSessionID INTEGER     NOT NULL REFERENCES ATTENDANCE_SESSION(AttendanceSessionID) ON DELETE CASCADE,
    AccountID           INTEGER     NOT NULL REFERENCES USER_ACCOUNT(AccountID) ON DELETE CASCADE,
    samples_total       INTEGER     NOT NULL DEFAULT 0,
    samples_analysed    INTEGER     NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (AttendanceSessionID, AccountID)
);

CREATE INDEX IF NOT EXISTS idx_behaviour_coverage_session
    ON BEHAVIOUR_COVERAGE(AttendanceSessionID);


-- ------------------------------------------------------------
-- 14. HEATMAP_SNAPSHOT  (U33 spatial activity heatmap data)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS HEATMAP_SNAPSHOT (
    HeatmapSnapshotID   SERIAL          PRIMARY KEY,
    AttendanceSessionID INTEGER         NOT NULL REFERENCES ATTENDANCE_SESSION(AttendanceSessionID) ON DELETE CASCADE,
    zone_x              INTEGER         NOT NULL,           -- grid cell column
    zone_y              INTEGER         NOT NULL,           -- grid cell row
    intensity           FLOAT           NOT NULL,           -- 0..1 normalised
    captured_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    camera_id           VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_heatmap_session_time
    ON HEATMAP_SNAPSHOT(AttendanceSessionID, captured_at);


-- ------------------------------------------------------------
-- 15. BEHAVIOUR_CONFIG  (U35 per-course toggle for behaviour analysis)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS BEHAVIOUR_CONFIG (
    CourseID            INTEGER         PRIMARY KEY REFERENCES COURSE(CourseID) ON DELETE CASCADE,
    enabled             BOOLEAN         NOT NULL DEFAULT FALSE,
    drowsiness          BOOLEAN         NOT NULL DEFAULT TRUE,
    phone_usage         BOOLEAN         NOT NULL DEFAULT TRUE,
    heatmap             BOOLEAN         NOT NULL DEFAULT TRUE,
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_by          INTEGER         REFERENCES USER_ACCOUNT(AccountID)
);

-- Per-course detection tuning (U35). All NULL by default, meaning "use the
-- server-wide AIConfig value"; an administrator overrides them per course
-- from the Behaviour Analysis Settings screen. Rooms differ enormously in
-- lighting and camera distance, and a single global threshold produced both
-- false alarms and silent misses during testing.
ALTER TABLE BEHAVIOUR_CONFIG
    ADD COLUMN IF NOT EXISTS ear_threshold          FLOAT
        CHECK (ear_threshold IS NULL OR ear_threshold BETWEEN 0.05 AND 0.60),
    ADD COLUMN IF NOT EXISTS mar_threshold          FLOAT
        CHECK (mar_threshold IS NULL OR mar_threshold BETWEEN 0.20 AND 1.50),
    ADD COLUMN IF NOT EXISTS headpose_pitch_deg     FLOAT
        CHECK (headpose_pitch_deg IS NULL OR headpose_pitch_deg BETWEEN 5 AND 89),
    ADD COLUMN IF NOT EXISTS phone_conf             FLOAT
        CHECK (phone_conf IS NULL OR phone_conf BETWEEN 0.05 AND 0.95),
    ADD COLUMN IF NOT EXISTS drowsy_confirm_seconds FLOAT
        CHECK (drowsy_confirm_seconds IS NULL OR drowsy_confirm_seconds BETWEEN 1 AND 120),
    -- Adaptive baseline: judge each student against their own open-eye EAR
    -- rather than a literature constant. Eye aperture varies far more between
    -- individuals than between alert and drowsy states for one person, so a
    -- fixed cut-off flags some students permanently and never flags others.
    ADD COLUMN IF NOT EXISTS adaptive_ear           BOOLEAN NOT NULL DEFAULT TRUE;


-- ------------------------------------------------------------
-- 16. ATTENDANCE_THRESHOLD_CONFIG  (U34 global reminder trigger)
-- ------------------------------------------------------------
-- Single-row table (id=1) holding the institution-wide settings.
CREATE TABLE IF NOT EXISTS ATTENDANCE_THRESHOLD_CONFIG (
    ConfigID                    INTEGER     PRIMARY KEY DEFAULT 1
                                            CHECK (ConfigID = 1),
    minimum_attendance_rate     FLOAT       NOT NULL DEFAULT 70.0
                                            CHECK (minimum_attendance_rate BETWEEN 0 AND 100),
    absence_threshold           INTEGER     NOT NULL DEFAULT 3
                                            CHECK (absence_threshold BETWEEN 1 AND 100),
    late_grace_seconds          INTEGER     NOT NULL DEFAULT 600,
    -- U03: default gap between detection windows during a scan (20 minutes).
    detection_interval_seconds  INTEGER     NOT NULL DEFAULT 1200
                                            CHECK (detection_interval_seconds BETWEEN 3 AND 86400),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by                  INTEGER     REFERENCES USER_ACCOUNT(AccountID)
);

-- Idempotent add for databases created before this column existed.
ALTER TABLE ATTENDANCE_THRESHOLD_CONFIG
    ADD COLUMN IF NOT EXISTS detection_interval_seconds INTEGER NOT NULL DEFAULT 1200;

INSERT INTO ATTENDANCE_THRESHOLD_CONFIG (ConfigID) VALUES (1)
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------------
-- 17. ATTENDANCE_RECORD.status — extend CHECK to allow new values
-- ------------------------------------------------------------
-- New statuses:
--   'leave'      → set by approveLeaveApplication (U31)
--   'early_left' → set by U16 early-departure flagging
ALTER TABLE ATTENDANCE_RECORD
    DROP CONSTRAINT IF EXISTS attendance_record_status_check;

ALTER TABLE ATTENDANCE_RECORD
    ADD CONSTRAINT attendance_record_status_check
    CHECK (status IN ('present', 'late', 'absent', 'leave', 'early_left'));


-- ------------------------------------------------------------
-- 18. SESSION_RECORDING  (U03 class recording, retained 30 days)
-- ------------------------------------------------------------
-- The full class video captured during a scan is stored locally and kept
-- for 30 days as appeal evidence, then automatically deleted. This table
-- tracks the recording's location and expiry; purge_expired_recordings()
-- removes expired rows (and their files) on startup.
CREATE TABLE IF NOT EXISTS SESSION_RECORDING (
    RecordingID         SERIAL          PRIMARY KEY,
    AttendanceSessionID INTEGER         NOT NULL UNIQUE
                                        REFERENCES ATTENDANCE_SESSION(AttendanceSessionID) ON DELETE CASCADE,
    file_path           TEXT,
    recorded_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ     NOT NULL DEFAULT (NOW() + INTERVAL '30 days')
);

CREATE INDEX IF NOT EXISTS idx_recording_expires ON SESSION_RECORDING(expires_at);


-- ------------------------------------------------------------
-- updated_at triggers for the new tables
-- ------------------------------------------------------------
DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'leave_application',
        'behaviour_config',
        'attendance_threshold_config'
    ] LOOP
        EXECUTE format('
            CREATE OR REPLACE TRIGGER trg_%s_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();
        ', t, t);
    END LOOP;
END;
$$;