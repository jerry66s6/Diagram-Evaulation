// Build config/study.json from a CSV manifest of diagrams.
//
//   node scripts/import-diagrams.mjs path/to/manifest.csv [options]
//
// Manifest columns (header row required, order free):
//   file              image path, relative to the manifest (.svg .png .jpg .jpeg .webp)
//   difficulty        easy | medium | hard
//   caption           the diagram's caption, shown next to it  } exactly one
//   caption_file      path to a .txt/.md file with the caption } of these two
//   id                optional stable ID; defaults to D-<first 10 hex of image SHA-256>
//   source            optional provenance, e.g. "arXiv:2401.01234 Fig. 3"
//
// Options:
//   --sample-size N   diagrams per participant, 20-30 (default: current value)
//   --fixed-order     every participant sees every diagram, in manifest row order;
//                     no sampling and no difficulty-balance check
//   --sets N          split the diagrams into N fixed sets with the same number of each
//                     difficulty (dealt in manifest order); each participant gets the
//                     next unused set, so no diagram goes to two participants
//   --mode M          demo | research (default: current value)
//   --id ID           catalog id; change it for each real study round
//   --title T         study title
//   --difficulty-note "How difficulty was assigned"
//   --out PATH        write somewhere other than config/study.json
//   --dry-run         validate and print the summary without writing anything
//
// Images are copied to public/diagrams/<first 20 hex of SHA-256>.<ext>. Nothing is
// deleted: keep old images while sessions from a previous catalog are still active.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, extname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { balancedAssignment } from '../lib/study-rules.mjs';

export const LEVELS = ['easy', 'medium', 'hard'];
const IMAGE_TYPES = new Set(['.svg', '.png', '.jpg', '.jpeg', '.webp']);
const ID_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

/** RFC 4180 CSV: quoted fields may contain commas, doubled quotes and line breaks. */
export function parseCsv(text) {
  const records = [];
  let record = [], field = '', quoted = false, fieldStarted = false;
  const input = text.replace(/^﻿/, '');
  const endField = () => { record.push(field); field = ''; fieldStarted = false; };
  const endRecord = () => { endField(); records.push(record); record = []; };
  for (let i = 0; i < input.length; i++) {
    const c = input[i];
    if (quoted) {
      if (c !== '"') field += c;
      else if (input[i + 1] === '"') { field += '"'; i++; }
      else quoted = false;
    } else if (c === '"' && !fieldStarted) { quoted = true; fieldStarted = true; }
    else if (c === ',') endField();
    else if (c === '\n' || c === '\r') { if (c === '\r' && input[i + 1] === '\n') i++; endRecord(); }
    else { field += c; fieldStarted = true; }
  }
  if (quoted) throw new Error('The manifest has an unterminated quoted field.');
  if (fieldStarted || record.length) endRecord();
  const rows = records.filter(r => r.some(value => value.trim() !== ''));
  if (!rows.length) throw new Error('The manifest is empty.');
  const header = rows[0].map(name => name.trim().toLowerCase());
  for (const required of ['file', 'difficulty']) {
    if (!header.includes(required)) throw new Error(`The manifest needs a "${required}" column.`);
  }
  if (!header.includes('caption') && !header.includes('caption_file')) {
    throw new Error('The manifest needs a "caption" or "caption_file" column.');
  }
  return rows.slice(1).map((values, n) => {
    if (values.length !== header.length) {
      throw new Error(`Manifest row ${n + 1} has ${values.length} columns; the header has ${header.length}.`);
    }
    return Object.fromEntries(header.map((name, i) => [name, values[i]]));
  });
}

/** Explains whether a pool can produce a balanced assignment of `sampleSize`. */
export function checkBalance(diagrams, sampleSize) {
  const counts = Object.fromEntries(LEVELS.map(level => [level, diagrams.filter(d => d.difficulty === level).length]));
  if (!Number.isInteger(sampleSize) || sampleSize < 20 || sampleSize > 30) {
    throw new Error(`Sample size must be an integer from 20 to 30; got ${sampleSize}.`);
  }
  try {
    balancedAssignment(diagrams, sampleSize, () => 0.5);
  } catch {
    const base = Math.floor(sampleSize / 3), extra = sampleSize % 3;
    const need = extra
      ? `at least ${base} per difficulty, with ${extra} of the groups having ${base + 1} or more`
      : `at least ${base} per difficulty`;
    throw new Error(`A ${sampleSize}-diagram assignment needs ${need}. The manifest has easy=${counts.easy}, medium=${counts.medium}, hard=${counts.hard}.`);
  }
  return counts;
}

/** Turns manifest rows into catalog entries plus the image copies needed. Reads files, writes nothing. */
export function buildCatalog(rows, { manifestDir, settings }) {
  const diagrams = [], copies = [], ids = new Map(), hashes = new Map();
  rows.forEach((row, n) => {
    const where = `Manifest row ${n + 1}`;
    const file = row.file?.trim();
    if (!file) throw new Error(`${where}: "file" is empty.`);
    const sourcePath = resolve(manifestDir, file);
    const extension = extname(sourcePath).toLowerCase();
    if (!IMAGE_TYPES.has(extension)) throw new Error(`${where}: ${file} is not an .svg, .png, .jpg or .webp image.`);
    if (!existsSync(sourcePath)) throw new Error(`${where}: image not found at ${sourcePath}.`);
    const sha256 = createHash('sha256').update(readFileSync(sourcePath)).digest('hex');
    if (hashes.has(sha256)) throw new Error(`${where}: ${file} has the same image content as manifest row ${hashes.get(sha256)}.`);
    hashes.set(sha256, n + 1);

    const difficulty = row.difficulty?.trim().toLowerCase();
    if (!LEVELS.includes(difficulty)) throw new Error(`${where}: difficulty must be easy, medium or hard; got "${row.difficulty ?? ''}".`);

    const inline = row.caption?.trim() ?? '', captionFile = row.caption_file?.trim() ?? '';
    if (inline && captionFile) throw new Error(`${where}: fill either caption or caption_file, not both.`);
    let caption = inline;
    if (captionFile) {
      const captionPath = resolve(manifestDir, captionFile);
      if (!existsSync(captionPath)) throw new Error(`${where}: caption file not found at ${captionPath}.`);
      caption = readFileSync(captionPath, 'utf8').trim();
    }
    if (!caption) throw new Error(`${where}: the caption is empty.`);

    const id = row.id?.trim() || `D-${sha256.slice(0, 10)}`;
    if (!ID_PATTERN.test(id)) throw new Error(`${where}: id "${id}" may only use letters, digits, ".", "_" and "-" (max 64).`);
    if (ids.has(id)) throw new Error(`${where}: id "${id}" is already used by manifest row ${ids.get(id)}.`);
    ids.set(id, n + 1);

    const imageName = `${sha256.slice(0, 20)}${extension === '.jpeg' ? '.jpg' : extension}`;
    copies.push({ from: sourcePath, name: imageName });
    diagrams.push({ id, image: `/diagrams/${imageName}`, caption, difficulty, source: row.source?.trim() || file, sha256 });
  });
  const catalog = { ...settings, diagrams };
  if (catalog.order === 'sets') {
    const sets = catalog.sets;
    const counts = Object.fromEntries(LEVELS.map(level => [level, diagrams.filter(d => d.difficulty === level).length]));
    if (!Number.isInteger(sets) || sets < 1) throw new Error('--sets must be a positive integer.');
    if (LEVELS.some(level => counts[level] % sets)) {
      throw new Error(`${sets} sets need each difficulty count to be a multiple of ${sets}; the manifest has easy=${counts.easy}, medium=${counts.medium}, hard=${counts.hard}.`);
    }
    for (const level of LEVELS) diagrams.filter(d => d.difficulty === level).forEach((d, i) => { d.set = i % sets + 1; });
    catalog.sampleSize = diagrams.length / sets;
    return { catalog, copies, counts };
  }
  if (catalog.order === 'fixed') {
    catalog.sampleSize = diagrams.length;
    const counts = Object.fromEntries(LEVELS.map(level => [level, diagrams.filter(d => d.difficulty === level).length]));
    return { catalog, copies, counts };
  }
  return { catalog, copies, counts: checkBalance(diagrams, catalog.sampleSize) };
}

function parseArgs(argv) {
  const options = { dryRun: false }, positional = [];
  const valued = { '--sample-size': 'sampleSize', '--sets': 'sets', '--mode': 'mode', '--id': 'id', '--title': 'title', '--difficulty-note': 'difficultyNote', '--out': 'out' };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dry-run') options.dryRun = true;
    else if (arg === '--fixed-order') options.fixedOrder = true;
    else if (arg === '--help' || arg === '-h') options.help = true;
    else if (valued[arg]) {
      if (i + 1 >= argv.length) throw new Error(`${arg} needs a value.`);
      options[valued[arg]] = argv[++i];
    } else if (arg.startsWith('--')) throw new Error(`Unknown option ${arg}.`);
    else positional.push(arg);
  }
  if (!options.help && positional.length !== 1) throw new Error('Pass exactly one manifest CSV path.');
  options.manifest = positional[0];
  return options;
}

function main() {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\nimport ')[0].replace(/^\/\/ ?/gm, ''));
    return;
  }
  const configPath = resolve(root, 'config/study.json');
  const current = existsSync(configPath) ? JSON.parse(readFileSync(configPath, 'utf8')) : {};
  const mode = options.mode ?? current.mode ?? 'demo';
  if (!['demo', 'research'].includes(mode)) throw new Error('--mode must be demo or research.');
  const settings = {
    id: options.id ?? current.id ?? 'diagram-study-v1',
    title: options.title ?? current.title ?? 'Diagram Evaluation Study',
    mode,
    sampleSize: options.sampleSize !== undefined ? Number(options.sampleSize) : current.sampleSize ?? 24,
    order: options.sets !== undefined ? 'sets' : options.fixedOrder ? 'fixed' : 'random',
    ...(options.sets !== undefined ? { sets: Number(options.sets) } : {}),
    difficultyNote: options.difficultyNote ?? current.difficultyNote ?? '',
  };
  if (options.sets !== undefined && options.fixedOrder) throw new Error('Use either --sets or --fixed-order, not both.');
  if (!ID_PATTERN.test(settings.id)) throw new Error('--id may only use letters, digits, ".", "_" and "-".');

  const manifestPath = resolve(options.manifest);
  const rows = parseCsv(readFileSync(manifestPath, 'utf8'));
  const { catalog, copies, counts } = buildCatalog(rows, { manifestDir: dirname(manifestPath), settings });

  const outPath = resolve(options.out ?? configPath);
  const imageDir = resolve(root, 'public/diagrams');
  const newImages = copies.filter(copy => !existsSync(resolve(imageDir, copy.name)));
  const summary = [
    `${catalog.diagrams.length} diagrams: easy=${counts.easy}, medium=${counts.medium}, hard=${counts.hard}.`,
    (catalog.order === 'sets'
      ? `${catalog.sets} sets of ${catalog.sampleSize} (${LEVELS.map(level => `${counts[level] / catalog.sets} ${level}`).join(', ')}); each participant gets the next unused set.`
      : catalog.order === 'fixed'
      ? `Every participant sees all ${catalog.diagrams.length}, in manifest order.`
      : `Each participant gets ${catalog.sampleSize}, balanced across difficulty.`) + ` Mode: ${catalog.mode}. Catalog id: ${catalog.id}.`,
    `${newImages.length} new image(s) for public/diagrams; ${copies.length - newImages.length} already present.`,
  ];
  if (options.dryRun) {
    console.log(['Dry run, nothing written.', ...summary].join('\n'));
    return;
  }
  mkdirSync(imageDir, { recursive: true });
  for (const copy of newImages) copyFileSync(copy.from, resolve(imageDir, copy.name));
  writeFileSync(outPath, JSON.stringify(catalog, null, 2) + '\n');
  const shown = relative(process.cwd(), outPath);
  console.log([`Wrote ${shown && !shown.startsWith('..') ? shown : outPath}.`, ...summary].join('\n'));
  if (catalog.mode === 'demo') console.log('Note: mode is "demo". Use --mode research for real participants.');
  console.log('Next: node scripts/validate-study.mjs, then rebuild or restart the dev server.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(`error: ${error.message}`); process.exit(1); }
}
