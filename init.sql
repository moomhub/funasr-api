-- FunASR Python API database initialization
-- Keep this file aligned with src/database/models.py.

CREATE DATABASE IF NOT EXISTS `funasr_tasks`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `funasr_tasks`;

CREATE TABLE IF NOT EXISTS `hotwords` (
  `id` INT NOT NULL AUTO_INCREMENT COMMENT '热词主键',
  `name` VARCHAR(100) NOT NULL COMMENT '热词名称，用于管理端展示',
  `text` TEXT NOT NULL COMMENT '热词内容，建议保存 JSON 数组，如 [{"weight":100,"hotword":"篮子"}]',
  `enabled` BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否启用',
  `is_deleted` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否软删除',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_hotwords_enabled_deleted` (`enabled`, `is_deleted`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='热词配置表';

CREATE TABLE IF NOT EXISTS `offline_tasks` (
  `id` VARCHAR(36) NOT NULL COMMENT 'OFFLINE 任务 ID/任务 key；新任务为 32 位无横线 UUID，兼容历史 36 位值',
  `filename` VARCHAR(255) NOT NULL COMMENT '原始上传文件名',
  `source_task_id` VARCHAR(36) NULL COMMENT '重识别来源 OFFLINE 任务 ID；普通上传任务为空',
  `file_size` INT NULL COMMENT '上传文件大小，单位 byte',
  `email` VARCHAR(255) NULL COMMENT '用户邮箱；为空时不发送完成消息',
  `vip` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否 VIP 优先任务',
  `hotwords` TEXT NULL COMMENT '本次任务传入的热词 JSON 数组字符串',
  `hotword_id` INT NULL COMMENT '热词表 ID；未传 hotwords 时可用此字段加载热词',
  `s3_key` VARCHAR(512) NULL COMMENT '归档 key；S3 为对象 key，本地为等价相对 key',
  `file_hash` VARCHAR(128) NULL COMMENT '原始文件 SHA-256，用于去重',
  `status` VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT '任务状态：pending/processing/completed/failed',
  `handle_status` TINYINT NOT NULL DEFAULT 2 COMMENT '任务处理标记状态',
  `is_deleted` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否软删除',
  `full_text` LONGTEXT NULL COMMENT '识别后的完整文本',
  `segments` LONGTEXT NULL COMMENT '句子/分段级识别结果 JSON',
  `word_timestamps` LONGTEXT NULL COMMENT '单字或词级时间戳 JSON，如 [["你",0,200],...]',
  `processing_time` FLOAT NULL COMMENT '处理耗时，单位秒',
  `error_message` TEXT NULL COMMENT '失败原因',
  `retry_count` INT NOT NULL DEFAULT 0 COMMENT '已重试次数',
  `max_retries` INT NOT NULL DEFAULT 3 COMMENT '最大重试次数',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `started_at` DATETIME NULL COMMENT '开始处理时间',
  `completed_at` DATETIME NULL COMMENT '完成或最终失败时间',
  PRIMARY KEY (`id`),
  KEY `idx_offline_tasks_status` (`status`),
  KEY `idx_offline_tasks_vip_created` (`vip`, `created_at`),
  KEY `idx_offline_tasks_created_at` (`created_at`),
  KEY `idx_offline_tasks_source_task_id` (`source_task_id`),
  KEY `idx_offline_tasks_s3_key` (`s3_key`),
  KEY `idx_offline_tasks_file_hash` (`file_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='OFFLINE ASR 任务表';

CREATE TABLE IF NOT EXISTS `spk_tasks` (
  `id` VARCHAR(36) NOT NULL COMMENT 'SPK 任务 ID/任务 key；新任务为 32 位无横线 UUID，兼容历史 36 位值',
  `filename` VARCHAR(255) NOT NULL COMMENT '原始上传文件名',
  `source_task_id` VARCHAR(36) NULL COMMENT '重识别来源 SPK 任务 ID；普通上传任务为空',
  `file_size` INT NULL COMMENT '上传文件大小，单位 byte',
  `email` VARCHAR(255) NULL COMMENT '用户邮箱；为空时不发送完成消息',
  `vip` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否 VIP 优先任务',
  `s3_key` VARCHAR(512) NULL COMMENT '归档 key；S3 为对象 key，本地为等价相对 key',
  `file_hash` VARCHAR(128) NULL COMMENT '原始文件 SHA-256，用于去重',
  `status` VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT '任务状态：pending/processing/completed/failed',
  `handle_status` TINYINT NOT NULL DEFAULT 2 COMMENT '任务处理标记状态',
  `is_deleted` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否软删除',
  `speaker_count` INT NOT NULL DEFAULT 0 COMMENT '识别出的说话人数量',
  `speaker_ids` JSON NULL COMMENT '说话人 ID 列表 JSON',
  `segments` JSON NULL COMMENT '说话人分段结果 JSON',
  `result` JSON NULL COMMENT '完整 SPK 返回结果 JSON',
  `processing_time` FLOAT NULL COMMENT '处理耗时，单位秒',
  `error_message` TEXT NULL COMMENT '失败原因',
  `retry_count` INT NOT NULL DEFAULT 0 COMMENT '已重试次数',
  `max_retries` INT NOT NULL DEFAULT 3 COMMENT '最大重试次数',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `started_at` DATETIME NULL COMMENT '开始处理时间',
  `completed_at` DATETIME NULL COMMENT '完成或最终失败时间',
  PRIMARY KEY (`id`),
  KEY `idx_spk_tasks_status` (`status`),
  KEY `idx_spk_tasks_vip_created` (`vip`, `created_at`),
  KEY `idx_spk_tasks_created_at` (`created_at`),
  KEY `idx_spk_tasks_source_task_id` (`source_task_id`),
  KEY `idx_spk_tasks_s3_key` (`s3_key`),
  KEY `idx_spk_tasks_file_hash` (`file_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='独立 SPK 任务表';

CREATE TABLE IF NOT EXISTS `s3_files` (
  `id` INT NOT NULL AUTO_INCREMENT COMMENT '文件索引主键',
  `task_key` VARCHAR(64) NOT NULL COMMENT '来源任务 key，对应 offline_tasks.id 或 spk_tasks.id',
  `task_type` VARCHAR(32) NOT NULL COMMENT '来源任务类型：offline/spk',
  `storage_backend` VARCHAR(16) NOT NULL DEFAULT 's3' COMMENT '实际存储后端：s3/local',
  `bucket_name` VARCHAR(255) NULL COMMENT 'S3 bucket 名称；本地存储时为空',
  `s3_key` VARCHAR(512) NOT NULL COMMENT '归档 key；S3 为对象 key，本地为等价相对 key',
  `original_filename` VARCHAR(255) NULL COMMENT '原始上传文件名',
  `stored_filename` VARCHAR(255) NOT NULL COMMENT '最终存储文件名，格式为任务 key + uuid + 后缀',
  `file_sha256` VARCHAR(128) NOT NULL COMMENT '原始文件 SHA-256，作为去重依据',
  `hash_algorithm` VARCHAR(32) NOT NULL DEFAULT 'sha256' COMMENT '文件 hash 算法',
  `file_size` INT NULL COMMENT '文件大小，单位 byte',
  `content_type` VARCHAR(100) NULL COMMENT '上传文件 MIME 类型',
  `local_path` VARCHAR(1024) NULL COMMENT '本地归档文件绝对或相对路径；S3 存储时为空',
  `upload_status` VARCHAR(20) NOT NULL DEFAULT 'uploaded' COMMENT '归档状态：uploaded/reused/failed',
  `is_reused` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否复用已存在的相同 hash 文件',
  `is_deleted` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否软删除',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '索引创建时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_s3_files_sha256` (`file_sha256`),
  KEY `idx_s3_files_task_key` (`task_key`),
  KEY `idx_s3_files_s3_key` (`s3_key`),
  KEY `idx_s3_files_backend` (`storage_backend`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='S3/本地归档文件索引表';
