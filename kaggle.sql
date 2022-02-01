CREATE TABLE `submissions` (
    `id` INTEGER PRIMARY KEY,
    `file_name` TEXT NOT NULL,
    `description` TEXT NOT NULL,
    `status` TEXT NOT NULL,
    `score` REAL,
    `running_time` REAL NOT NULL
)