CREATE TABLE `study_ratings` (
	`session_id` text NOT NULL,
	`diagram_id` text NOT NULL,
	`scores` text NOT NULL,
	`comment` text DEFAULT '' NOT NULL,
	`duration_ms` integer DEFAULT 0 NOT NULL,
	`complete` integer DEFAULT 0 NOT NULL,
	`updated_at` text NOT NULL,
	PRIMARY KEY(`session_id`, `diagram_id`),
	FOREIGN KEY (`session_id`) REFERENCES `study_sessions`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `study_sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`token_hash` text NOT NULL,
	`participant_code` text NOT NULL,
	`study_version` text NOT NULL,
	`mode` text NOT NULL,
	`assignment` text NOT NULL,
	`consent_version` text NOT NULL,
	`started_at` text NOT NULL,
	`submitted_at` text
);
--> statement-breakpoint
CREATE UNIQUE INDEX `study_sessions_token_hash_unique` ON `study_sessions` (`token_hash`);--> statement-breakpoint
CREATE UNIQUE INDEX `study_sessions_participant_code_unique` ON `study_sessions` (`participant_code`);--> statement-breakpoint
CREATE INDEX `idx_sessions_version` ON `study_sessions` (`study_version`);