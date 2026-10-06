// Deploy the study to Cloudflare Workers + D1 with one command.
//
//   npx wrangler login              # once, opens the browser
//   npm run deploy:cloudflare       # creates or reuses everything, then deploys
//   npm run deploy:cloudflare -- --dry-run   # build and check only; no login needed
//   npm run deploy:cloudflare -- --rotate-key # also replace the researcher key
//
// Every step is safe to re-run:
//   1. validate config/study.json and its images
//   2. find or create the D1 database named in config/cloudflare.json (id saved back)
//   3. build with the real Worker name and database id
//   4. apply new SQL migrations from drizzle/ to the remote database
//   5. deploy the Worker
//   6. set the STUDY_ADMIN_KEY secret if missing; a new key is saved to .env.cloudflare
import { spawnSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { chmodSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const cloudflareConfigPath = resolve(root, 'config/cloudflare.json');
const keyFile = resolve(root, '.env.cloudflare');
const builtConfig = resolve(root, 'dist/server/wrangler.json');
const dryRun = process.argv.includes('--dry-run');
const rotateKey = process.argv.includes('--rotate-key');

function step(message) { console.log(`\n▸ ${message}`); }
function fail(message) { console.error(`\nerror: ${message}`); process.exit(1); }

function run(args, { capture = false, input, env = {} } = {}) {
  const result = spawnSync(process.execPath, args, {
    cwd: root,
    encoding: 'utf8',
    input,
    env: { ...process.env, WRANGLER_SEND_METRICS: 'false', ...env },
    stdio: capture || input !== undefined ? ['pipe', 'pipe', 'pipe'] : 'inherit',
  });
  if (result.error) fail(result.error.message);
  return result;
}
const wrangler = (args, options) =>
  run(['--import', './scripts/sites-env.mjs', './node_modules/wrangler/bin/wrangler.js', ...args], options);

function readCloudflareConfig() {
  const config = JSON.parse(readFileSync(cloudflareConfigPath, 'utf8'));
  for (const key of ['workerName', 'databaseName']) {
    if (!/^[a-z0-9][a-z0-9-]{0,62}$/.test(config[key] ?? '')) {
      fail(`config/cloudflare.json: ${key} must use lowercase letters, digits and dashes.`);
    }
  }
  return config;
}

function findDatabaseId(name) {
  const listed = wrangler(['d1', 'list', '--json'], { capture: true });
  if (listed.status !== 0) fail(`could not list D1 databases.\n${listed.stderr || listed.stdout}`);
  const start = listed.stdout.indexOf('[');
  const databases = start >= 0 ? JSON.parse(listed.stdout.slice(start)) : [];
  const match = databases.find(db => db.name === name);
  return match ? (match.uuid ?? match.id ?? '') : '';
}

function readSavedKey() {
  if (!existsSync(keyFile)) return '';
  const line = readFileSync(keyFile, 'utf8').split('\n').find(l => l.startsWith('STUDY_ADMIN_KEY='));
  return line ? line.slice('STUDY_ADMIN_KEY='.length).trim() : '';
}

const config = readCloudflareConfig();

step('Checking the diagram catalog');
if (run(['scripts/validate-study.mjs']).status !== 0) fail('fix config/study.json before deploying.');
const study = JSON.parse(readFileSync(resolve(root, 'config/study.json'), 'utf8'));
if (study.mode !== 'research') console.log(`  Note: the catalog is in "${study.mode}" mode. Import real diagrams with --mode research before recruiting.`);

let databaseId = config.databaseId;
if (dryRun) {
  databaseId ||= '00000000-0000-4000-8000-000000000000';
} else {
  step('Checking Cloudflare login');
  const who = wrangler(['whoami'], { capture: true });
  if (who.status !== 0 || /not authenticated/i.test(who.stdout + who.stderr)) {
    fail('not logged in to Cloudflare. Run `npx wrangler login` in user-study/, then run this again.');
  }
  const account = (who.stdout.match(/associated with the email (\S+@\S+?)\.?(?:\s|$)/) || [])[1];
  console.log(`  Logged in${account ? ` as ${account}` : ''}.`);

  step(`Finding D1 database "${config.databaseName}"`);
  if (!databaseId) databaseId = findDatabaseId(config.databaseName);
  if (!databaseId) {
    console.log('  Not found; creating it.');
    const created = wrangler(['d1', 'create', config.databaseName]);
    if (created.status !== 0) fail('could not create the D1 database.');
    databaseId = findDatabaseId(config.databaseName);
    if (!databaseId) fail('the database was created but its id could not be read. Run this script again.');
  }
  if (databaseId !== config.databaseId) {
    writeFileSync(cloudflareConfigPath, JSON.stringify({ ...config, databaseId }, null, 2) + '\n');
    console.log('  Saved its id to config/cloudflare.json.');
  }
  console.log(`  Using database ${databaseId}.`);
}

step('Building the site');
const built = run(['scripts/run-framework.mjs', 'build'], {
  env: { CF_WORKER_NAME: config.workerName, CF_D1_DATABASE_NAME: config.databaseName, CF_D1_DATABASE_ID: databaseId },
});
if (built.status !== 0) fail('the build failed.');
const output = JSON.parse(readFileSync(builtConfig, 'utf8'));
const binding = output.d1_databases?.find(db => db.binding === 'DB');
if (output.name !== config.workerName || binding?.database_id !== databaseId || !binding?.migrations_dir) {
  fail('the built Worker config does not contain the expected name and database. Check vite.config.ts.');
}

if (dryRun) {
  step('Checking the Worker bundle (dry run, nothing is uploaded)');
  const checked = wrangler(['deploy', '--dry-run', '--config', builtConfig]);
  if (checked.status !== 0) fail('the dry run failed.');
  console.log('\nDry run passed. Run `npx wrangler login`, then `npm run deploy:cloudflare` to deploy.');
  process.exit(0);
}

step('Applying database migrations');
const migrated = wrangler(['d1', 'migrations', 'apply', config.databaseName, '--remote', '--config', builtConfig]);
if (migrated.status !== 0) fail('migrations failed; nothing was deployed.');

step('Deploying the Worker');
const deployed = wrangler(['deploy', '--config', builtConfig], { capture: true });
process.stdout.write(deployed.stdout);
process.stderr.write(deployed.stderr);
if (deployed.status !== 0) {
  if (/workers\.dev subdomain/i.test(deployed.stdout + deployed.stderr)) {
    fail('register a workers.dev subdomain first: Cloudflare dashboard → Workers & Pages → your subdomain. Then run this again.');
  }
  fail('deploy failed.');
}
const url = (deployed.stdout.match(/https:\/\/[^\s]+\.workers\.dev/) || [])[0];

step('Checking the researcher key');
const secrets = wrangler(['secret', 'list', '--config', builtConfig, '--format', 'json'], { capture: true });
if (secrets.status !== 0) fail(`could not list Worker secrets.\n${secrets.stderr || secrets.stdout}`);
if (/"STUDY_ADMIN_KEY"/.test(secrets.stdout) && !rotateKey) {
  console.log(existsSync(keyFile)
    ? '  STUDY_ADMIN_KEY is already set; your copy is in .env.cloudflare.'
    : '  STUDY_ADMIN_KEY is already set, but .env.cloudflare is missing here. Run with --rotate-key to make a new one.');
} else {
  const key = (!rotateKey && readSavedKey()) || randomBytes(24).toString('base64url');
  writeFileSync(keyFile, `# Researcher key for the deployed study. Keep private; never commit.\nSTUDY_ADMIN_KEY=${key}\n`);
  chmodSync(keyFile, 0o600);
  const put = wrangler(['secret', 'put', 'STUDY_ADMIN_KEY', '--config', builtConfig], { input: key });
  if (put.status !== 0) fail(`could not set the secret.\n${put.stderr || put.stdout}`);
  console.log('  Set STUDY_ADMIN_KEY. Your copy is in user-study/.env.cloudflare.');
}

console.log(`
Done.
  Participant link:     ${url ?? '(see the deploy output above)'}
  Researcher workspace: ${url ? url + '/researcher' : '<link>/researcher'}
  Researcher key:       user-study/.env.cloudflare
To update after changing diagrams or code, run npm run deploy:cloudflare again.`);
