DROP TABLE IF EXISTS role;

CREATE TABLE IF NOT EXISTS role (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT NOT NULL
);

DROP TABLE IF EXISTS users;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT,
    lang TEXT NOT NULL,
    role_id INTEGER,
    FOREIGN KEY (role_id) REFERENCES role(id)
);

DROP TABLE IF EXISTS user_details;

CREATE TABLE IF NOT EXISTS user_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    child_name TEXT NOT NULL,
    father_name TEXT,
    mother_name TEXT,
    phone_number TEXT,
    gifts TEXT NOT NULL,
    context TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

DROP TABLE IF EXISTS calls;

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    call_date DATE,
    call_time TIME,
    time_zone TEXT,
    verification_code TEXT,
    call_job_id TEXT,
    timer INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

DROP TABLE IF EXISTS discounts;
CREATE TABLE IF NOT EXISTS discounts (
    code TEXT PRIMARY KEY,
    discount_value REAL,
    active BOOLEAN,
    validity_date DATE, 
    usage_count INTEGER, 
    unlimited_usage BOOLEAN DEFAULT FALSE, 
    unlimited_validity BOOLEAN DEFAULT FALSE,
    created_by_user_id INTEGER,
    FOREIGN KEY (created_by_user_id) REFERENCES users(id)
);

INSERT INTO discounts (code, discount_value, active, unlimited_usage, unlimited_validity) 
VALUES ('TEST100', 100, TRUE, TRUE, TRUE);

INSERT INTO role (role_name)
VALUES ('admin');
INSERT INTO role (role_name)
VALUES ('user');