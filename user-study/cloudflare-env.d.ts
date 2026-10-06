declare namespace Cloudflare {
  interface Env {
    DB?: D1Database;
    STUDY_ADMIN_KEY?: string;
    BUCKET?: R2Bucket;
  }
}
