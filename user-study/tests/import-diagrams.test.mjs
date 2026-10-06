import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildCatalog, checkBalance, parseCsv } from '../scripts/import-diagrams.mjs';

const settings = { id: 'test-study', title: 'Test', mode: 'demo', sampleSize: 24, difficultyNote: '' };

function fixture(perLevel = { easy: 8, medium: 8, hard: 8 }) {
  const dir = mkdtempSync(join(tmpdir(), 'import-diagrams-'));
  mkdirSync(join(dir, 'images'));
  const rows = [];
  for (const [difficulty, count] of Object.entries(perLevel)) {
    for (let i = 0; i < count; i++) {
      const file = `images/${difficulty}-${i}.svg`;
      writeFileSync(join(dir, file), `<svg xmlns="http://www.w3.org/2000/svg"><text>${difficulty} ${i}</text></svg>`);
      rows.push({ file, difficulty, caption: `The ${difficulty} flow number ${i}.`, source: `fixture ${difficulty} ${i}` });
    }
  }
  return { dir, rows };
}

test('CSV parsing handles quotes, commas, line breaks, CRLF and a BOM', () => {
  const rows = parseCsv('﻿id,file,difficulty,caption\r\nA1,a.svg,easy,"Start, then ""Validate""\nthen End"\r\nA2,b.png,Hard,Plain\r\n\r\n');
  assert.equal(rows.length, 2);
  assert.equal(rows[0].caption, 'Start, then "Validate"\nthen End');
  assert.equal(rows[1].difficulty, 'Hard');
});

test('CSV parsing rejects a missing column, a ragged row and an open quote', () => {
  assert.throws(() => parseCsv('file,caption\na.svg,x\n'), /difficulty/);
  assert.throws(() => parseCsv('file,difficulty,caption\na.svg,easy\n'), /row 1 has 2 columns/);
  assert.throws(() => parseCsv('file,difficulty,caption\na.svg,easy,"open\n'), /unterminated/);
});

test('a balanced manifest becomes a catalog with hash-named images and stable default IDs', () => {
  const { dir, rows } = fixture();
  rows[0].id = 'FC001';
  rows[1].difficulty = ' EASY ';
  const { catalog, copies, counts } = buildCatalog(rows, { manifestDir: dir, settings });
  assert.deepEqual(counts, { easy: 8, medium: 8, hard: 8 });
  assert.equal(catalog.diagrams.length, 24);
  assert.equal(catalog.diagrams[0].id, 'FC001');
  assert.equal(catalog.diagrams[1].difficulty, 'easy');
  const second = catalog.diagrams[2];
  const expected = createHash('sha256').update(`<svg xmlns="http://www.w3.org/2000/svg"><text>easy 2</text></svg>`).digest('hex');
  assert.equal(second.sha256, expected);
  assert.equal(second.id, `D-${expected.slice(0, 10)}`);
  assert.equal(second.image, `/diagrams/${expected.slice(0, 20)}.svg`);
  assert.equal(copies.length, 24);
  assert.equal(catalog.sampleSize, 24);
});

test('captions can come from a file but not from both places', () => {
  const { dir, rows } = fixture();
  writeFileSync(join(dir, 'long.md'), '  Line one.\nLine two.  \n');
  rows[0] = { ...rows[0], caption: '', caption_file: 'long.md' };
  assert.equal(buildCatalog(rows, { manifestDir: dir, settings }).catalog.diagrams[0].caption, 'Line one.\nLine two.');
  rows[0].caption = 'Also inline';
  assert.throws(() => buildCatalog(rows, { manifestDir: dir, settings }), /either caption or caption_file/);
});

test('bad rows are rejected with the row number', () => {
  const cases = [
    [rows => { rows[3].difficulty = 'tricky'; }, /row 4: difficulty/],
    [rows => { rows[2].file = 'images/missing.svg'; }, /row 3: image not found/],
    [rows => { rows[5].file = rows[4].file; }, /row 6: .* same image content as manifest row 5/],
    [rows => { rows[1].id = 'X'; rows[7].id = 'X'; }, /row 8: id "X" is already used/],
    [rows => { rows[0].id = 'has space'; }, /row 1: id/],
    [rows => { rows[0].caption = '   '; }, /row 1: the caption is empty/],
    [rows => { rows[0].file = 'notes.txt'; }, /row 1: notes\.txt is not/],
  ];
  for (const [mutate, message] of cases) {
    const { dir, rows } = fixture();
    writeFileSync(join(dir, 'notes.txt'), 'x');
    mutate(rows);
    assert.throws(() => buildCatalog(rows, { manifestDir: dir, settings }), message);
  }
});

test('an unbalanced pool explains what is missing', () => {
  const { dir, rows } = fixture({ easy: 12, medium: 10, hard: 6 });
  assert.throws(() => buildCatalog(rows, { manifestDir: dir, settings }), /at least 8 per difficulty.*hard=6/);
  const pool = rows.map(r => ({ id: r.file, difficulty: r.difficulty }));
  assert.deepEqual(checkBalance(pool, 20), { easy: 12, medium: 10, hard: 6 });
  assert.throws(() => checkBalance(pool, 31), /20 to 30/);
});
