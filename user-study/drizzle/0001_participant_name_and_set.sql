ALTER TABLE `study_sessions` ADD `participant_name` text DEFAULT '' NOT NULL;--> statement-breakpoint
ALTER TABLE `study_sessions` ADD `set_number` integer;--> statement-breakpoint
CREATE UNIQUE INDEX `study_sessions_version_set_unique` ON `study_sessions` (`study_version`,`set_number`);
