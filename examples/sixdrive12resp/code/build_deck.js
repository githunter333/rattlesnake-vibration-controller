// Control-law comparison deck, built from the same JSON the report uses.
const pptxgen = require('pptxgenjs');
const fs = require('fs');

const R = JSON.parse(fs.readFileSync('report_data.json'));
const CT = JSON.parse(fs.readFileSync('crossterm_data.json'));
const SV = JSON.parse(fs.readFileSync('singular_value_data.json'));
const by = Object.fromEntries(R.map(r => [r.run, r]));

// Charcoal Minimal, with the figures' own orange as the sharp accent so the
// deck and the plots read as one system.
const INK = '36454F', INK2 = '5E6B73', MUTED = '8A959B';
const BG = 'FFFFFF', PANEL = 'F2F2F2', DARK = '2B3740';
const ORANGE = 'EB6834', BLUE = '2A78D6';
const HEAD = 'Cambria', BODY = 'Calibri';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';            // 13.3 x 7.5
pres.author = 'nhunter';
pres.title = 'Control-law comparison on the 6-12-8 system';
const W = 13.3, H = 7.5, M = 0.6;

const NICE = {
  match_trace: 'match_trace_pseudoinverse',
  optimal_diagonal: 'optimal_diagonal_control',
  optimal_diagonal_fast: 'optimal_diagonal_control_fast',
  congruence: 'match_diagonal_congruence',
  pseudoinverse: 'pseudoinverse_control',
  buzz: 'buzz_control',
};

function titleSlide(s, n, title, kicker) {
  s.addText(title, {
    x: M, y: 0.34, w: W - 2 * M - 0.8, h: 0.72, isTextBox: true,
    fontFace: HEAD, fontSize: 30, bold: true, color: INK, margin: 0,
  });
  if (kicker) {
    s.addText(kicker, {
      x: M, y: 1.06, w: W - 2 * M - 0.8, h: 0.42, isTextBox: true,
      fontFace: BODY, fontSize: 15, color: INK2, margin: 0,
    });
  }
  // motif: the slide number in a filled circle, same spot on every slide
  s.addShape(pres.ShapeType.ellipse, {
    x: W - M - 0.46, y: 0.38, w: 0.46, h: 0.46, fill: { color: PANEL },
  });
  s.addText(String(n), {
    x: W - M - 0.46, y: 0.38, w: 0.46, h: 0.46, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: INK2,
    align: 'center', valign: 'middle', margin: 0,
  });
}

function caption(s, text, y) {
  s.addText(text, {
    x: M, y: y, w: W - 2 * M, h: 0.5, isTextBox: true,
    fontFace: BODY, fontSize: 11.5, color: MUTED, margin: 0,
  });
}

// fit an image inside a box, preserving aspect, centred
// Fit inside the box preserving aspect. Left-aligned by default so charts
// share the content column's left edge with the tables and tiles below them;
// centring made the two bands visibly disagree.
function fitImage(s, path, ar, box, centre) {
  let w = box.w, h = w / ar;
  if (h > box.h) { h = box.h; w = h * ar; }
  const x = centre ? box.x + (box.w - w) / 2 : box.x;
  s.addImage({ path, x, y: box.y + (box.h - h) / 2, w, h });
}

function statTile(s, x, y, w, value, label, color) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 1.5, fill: { color: PANEL }, rectRadius: 0.06,
  });
  s.addText(value, {
    x: x + 0.16, y: y + 0.16, w: w - 0.32, h: 0.68, isTextBox: true,
    fontFace: HEAD, fontSize: 30, bold: true, color: color || INK,
    align: 'center', valign: 'middle', margin: 0,
  });
  s.addText(label, {
    x: x + 0.16, y: y + 0.84, w: w - 0.32, h: 0.54, isTextBox: true,
    fontFace: BODY, fontSize: 11, color: INK2,
    align: 'center', valign: 'top', margin: 0,
  });
}

/* ---------------------------------------------------------- 1. title */
let s = pres.addSlide();
s.background = { color: DARK };
s.addText('Control-law comparison', {
  x: M, y: 2.0, w: W - 2 * M, h: 0.8, isTextBox: true,
  fontFace: HEAD, fontSize: 44, bold: true, color: 'FFFFFF', margin: 0,
});
s.addText('on the 6-12-8 system', {
  x: M, y: 2.82, w: W - 2 * M, h: 0.7, isTextBox: true,
  fontFace: HEAD, fontSize: 34, color: ORANGE, margin: 0,
});
s.addText(
  'Twelve runs  ·  six laws  ·  drive-coherence cap on and off  ·  linear plant',
  { x: M, y: 3.86, w: W - 2 * M, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 16, color: 'D6DDE2', margin: 0 });
s.addText([
  { text: 'rattlesnake-vibration-controller', options: { breakLine: true } },
  { text: 'branch control-law-development-2026-09  ·  15 September 2026' },
], { x: M, y: 5.6, w: W - 2 * M, h: 0.8, isTextBox: true,
     fontFace: BODY, fontSize: 12, color: '9FAEB8', margin: 0 });
s.addNotes(
  'Twelve runs on one linear plant with every other setting held fixed and '
  + 'machine-verified across 23 settings. Six laws, each with the drive-coherence '
  + 'cap on (0.95) and off (1.0). Runs 01-08 are the four closed-loop laws; '
  + 'runs 09-12 added pseudoinverse_control and buzz_control, the two original '
  + 'upstream open-loop laws.\n\n'
  + 'Three findings drive the deck: the coherence cap means opposite things to '
  + 'feedback and open-loop laws; a law with no error feedback finishes within '
  + '0.7 dB of the best feedback law; and the plant is rank 5, not 6, which '
  + 'makes part of the specification unreachable by construction.');

/* ------------------------------------------------- 2. system under test */
s = pres.addSlide();
titleSlide(s, 2, 'The system under test',
  'Six drives, twelve responses, eight control channels, 100–1000 Hz');
fitImage(s, 'figs/deck_system.png', 2.277,
  { x: M, y: 1.52, w: W - 2 * M, h: 2.98 });
// the chart is height-limited, so fill the column beside it rather than
// leaving 5 in of white
s.addShape(pres.ShapeType.roundRect, {
  x: M + 7.1, y: 1.52, w: W - 2 * M - 7.1, h: 2.98,
  fill: { color: PANEL }, rectRadius: 0.06,
});
s.addText([
  { text: 'Six drives in, eight channels controlled\n',
    options: { bold: true, fontSize: 16, color: INK, breakLine: true } },
  { text: 'Twelve responses are measured; the first eight are the control set '
        + '(nodes 7–14, X direction). All six shakers drive.\n\n',
    options: { fontSize: 13, color: INK2, breakLine: true } },
  { text: 'But only five directions exist\n',
    options: { bold: true, fontSize: 16, color: ORANGE, breakLine: true } },
  { text: 'The FRF is rank 5, so part of the flat specification is unreachable '
        + 'by any law — before control design enters the picture at all. '
        + 'Slide 3 has the evidence.',
    options: { fontSize: 13, color: INK2 } },
], { x: M + 7.3, y: 1.7, w: W - 2 * M - 7.5, h: 2.66, isTextBox: true,
     fontFace: BODY, margin: 0, valign: 'top' });

statTile(s, M, 4.93, 2.72, '33', 'modes in the control band');
statTile(s, M + 2.92, 4.93, 2.72, '0.83 %', 'median modal damping\n(0.70–1.19 %)');
statTile(s, M + 5.84, 4.93, 2.72, '4096 Hz', 'sample rate, virtual\nhardware (type 6)');
statTile(s, M + 8.76, 4.93, 2.72, '1×10⁻³', 'flat spec, zero\ncross-terms', ORANGE);
caption(s, 'Linear SDynPy frame model sdynpy_frame6x12_system.npz, 36 DOF, run as virtual hardware.', 6.62);
s.addNotes(
  'Lightly damped structure — median zeta 0.83 %, minimum 0.70 % — which is the '
  + 'harder case for control. A better-damped twin exists '
  + '(sdynpy_frame6x12_system_higherz.npz, identical mass and stiffness, zeta raised '
  + 'to a 3.34 % median); its reachability floor is 4.05 dB against this plant\'s '
  + '4.09, so damping does not change what is reachable — it halves the conditioning.\n\n'
  + 'Left panel: modal damping histogram. Centre: reachability floor per run at the '
  + 'rcond the laws use, against unrestricted over the real rank-5 subspace — they '
  + 'agree, which is the point. Right: median cond(H) per run, scattering by a factor '
  + 'of ten across identical identifications of a deterministic plant. See slide 3 for '
  + 'why that scatter is an artefact.');

/* -------------------------------------------- 3. singular values / rank 5 */
s = pres.addSlide();
titleSlide(s, 3, 'The plant is rank 5, not 6',
  'Six shakers span a five-dimensional response subspace at these eight locations');
fitImage(s, 'figs/deck_singular_values.png', 2.388,
  { x: M, y: 1.48, w: W - 2 * M, h: 3.34 });

const svRows = [[
  { text: '', options: { fill: { color: PANEL } } },
  ...['σ1', 'σ2', 'σ3', 'σ4', 'σ5', 'σ6'].map(t => ({
    text: t, options: { bold: true, color: INK, fill: { color: PANEL }, align: 'center' } })),
]];
svRows.push([
  { text: 'analytic, dB re σ1', options: { color: INK2 } },
  ...SV.analytic.map((a, i) => ({
    text: a.i === 6 ? '−327' : a.db.toFixed(1),
    options: { align: 'center', bold: a.i === 6, color: a.i === 6 ? ORANGE : INK } })),
]);
svRows.push([
  { text: 'measured, dB re σ1', options: { color: INK2 } },
  ...SV.measured.map((m, i) => ({
    text: m.db.toFixed(1),
    options: { align: 'center', bold: m.i === 6, color: m.i === 6 ? ORANGE : INK } })),
]);
s.addTable(svRows, {
  x: M, y: 5.06, w: 7.7, fontFace: BODY, fontSize: 11.5, color: INK,
  border: { type: 'solid', color: 'DDE1E3', pt: 0.5 },
  colW: [1.9, 0.966, 0.966, 0.966, 0.966, 0.966, 0.966], rowH: 0.3,
});
s.addShape(pres.ShapeType.roundRect, {
  x: M + 8.0, y: 5.06, w: W - 2 * M - 8.0, h: 1.52,
  fill: { color: PANEL }, rectRadius: 0.06,
});
s.addText([
  { text: 'Quote σ1/σ5, not σ1/σ6\n', options: { bold: true, fontSize: 13, color: INK } },
  { text: 'Real conditioning is 68 (analytic), 64 (measured). '
        + 'The cond(H) column in the report is σ1/σ6 ≈ 3567 — that is the '
        + 'identification noise floor, which is why it scatters ten-fold.',
    options: { fontSize: 11.5, color: INK2 } },
], { x: M + 8.2, y: 5.20, w: W - 2 * M - 8.4, h: 1.26, isTextBox: true,
     fontFace: BODY, margin: 0, valign: 'top' });
s.addNotes(
  'Computed exactly from the model mass, stiffness and damping matrices. '
  + 'Validated against the measured FRF: magnitude agrees to a median 0.21 dB and '
  + 'sigma1..sigma5 reproduce to a few tenths of a dB. Sigma6 is ~1e-16 analytically '
  + '— machine round-off on a 36-DOF solve — and 2.4e-3 measured, which is '
  + 'identification noise, not a sixth direction. Numeric rank is 5 in all twelve runs.\n\n'
  + 'Two consequences. First, rcond = 1e-3 is doing the right thing by truncating that '
  + 'direction; an earlier draft of the report claimed 1.6 dB of headroom was being lost '
  + 'to the truncation, which was wrong and has been withdrawn. Second, the sixth drive '
  + 'direction produces no response at these eight locations — worth checking whether '
  + 'that is by construction, because on a real rig it would mean a shaker doing nothing.');

/* -------------------------------------------------------- 4. what was run */
s = pres.addSlide();
titleSlide(s, 4, 'What was run',
  'One plant, one configuration, twelve runs — comparability checked by machine');

const lawRows = [
  ['match_trace_pseudoinverse', 'feedback', '01 / 02'],
  ['optimal_diagonal_control', 'feedback', '03 / 04'],
  ['optimal_diagonal_control_fast', 'feedback', '05 / 06'],
  ['match_diagonal_congruence', 'feedback', '07 / 08'],
  ['pseudoinverse_control', 'open-loop', '09 / 10'],
  ['buzz_control', 'open-loop', '11 / 12'],
];
const t4 = [[
  { text: 'Control law', options: { bold: true, color: INK, fill: { color: PANEL } } },
  { text: 'Loop', options: { bold: true, color: INK, fill: { color: PANEL } } },
  { text: 'Runs (cap on / off)', options: { bold: true, color: INK, fill: { color: PANEL } } },
]];
lawRows.forEach(r => t4.push([
  { text: r[0], options: { fontFace: 'Courier New', fontSize: 11 } },
  { text: r[1], options: { color: r[1] === 'open-loop' ? ORANGE : BLUE, bold: true } },
  { text: r[2], options: { color: INK2 } },
]));
s.addTable(t4, {
  x: M, y: 1.72, w: 7.2, fontFace: BODY, fontSize: 12, color: INK,
  border: { type: 'solid', color: 'DDE1E3', pt: 0.5 },
  colW: [4.0, 1.5, 1.7], rowH: 0.36,
});

s.addShape(pres.ShapeType.roundRect, {
  x: M + 7.6, y: 1.72, w: W - 2 * M - 7.6, h: 2.42,
  fill: { color: PANEL }, rectRadius: 0.06,
});
s.addText([
  { text: 'Held fixed across all twelve\n', options: { bold: true, fontSize: 13, color: INK, breakLine: true } },
  { text: 'samples per frame    4096\nsample rate          4096 Hz\n'
        + 'frames in CPSD       20\ncontrol averaging    Exponential 0.04\n'
        + 'update sys ID        Y\nsys ID averages      20\n'
        + 'sys ID level         1.0 V rms\nrcond                1e-3',
    options: { fontSize: 11, color: INK2, fontFace: 'Courier New' } },
], { x: M + 7.8, y: 1.88, w: W - 2 * M - 8.0, h: 2.6, isTextBox: true,
     fontFace: BODY, margin: 0, valign: 'top' });

s.addText([
  { text: '23 settings are compared across runs by ', options: { color: INK2 } },
  { text: 'score_runs_sdynpy.py', options: { fontFace: 'Courier New', color: INK } },
  { text: '’s MUST_MATCH list. A mismatch raises a NOT-COMPARABLE warning rather '
        + 'than being scored silently — a response to an earlier campaign where sixteen '
        + 'runs turned out to span three different plants.', options: { color: INK2 } },
], { x: M, y: 4.92, w: W - 2 * M, h: 0.9, isTextBox: true,
     fontFace: BODY, fontSize: 13, margin: 0 });
s.addNotes(
  'Every run used the same plant, sample rate, averaging and system-ID configuration. '
  + 'The MUST_MATCH provenance check exists because of a real failure: an earlier '
  + 'sixteen-run set was scored as one comparison and turned out to span linear, '
  + 'shifted and nonlinear plants with differing update_tf_during_control.\n\n'
  + 'Cap on means max_drive_coherence = 0.95; cap off means 1.0, which is a no-op. '
  + 'Note the cap sits in parameter position 2 for four of the laws but position 5 for '
  + 'the two optimal-diagonal laws — an easy thing to get wrong when reading profiles.');

/* ------------------------------------------------------------ 5. results */
s = pres.addSlide();
titleSlide(s, 5, 'Results', 'Scored against the flat specification, and against what the plant can deliver');

const hdr = ['Run', 'Law', 'Loop', 'Cap', 'rms dB', '% out', 'vs floor', 'drive V²', 'level dB'];
const t5 = [hdr.map(h => ({ text: h, options: { bold: true, color: INK, fill: { color: PANEL }, align: h === 'Run' || h === 'Law' || h === 'Loop' || h === 'Cap' ? 'left' : 'right' } }))];
const bestRms = Math.min(...R.map(r => r.rms));
const bestOut = Math.min(...R.map(r => r.pout));
const bestAch = Math.min(...R.map(r => r.ach_rms));
R.slice().sort((a, b) => a.run.localeCompare(b.run)).forEach(r => {
  const open = r.loop === 'open';
  t5.push([
    { text: r.run, options: { fontFace: 'Courier New', fontSize: 10 } },
    { text: NICE[r.law], options: { fontFace: 'Courier New', fontSize: 9.5 } },
    { text: open ? 'open' : 'fbk', options: { color: open ? ORANGE : BLUE, bold: true } },
    { text: r.cap },
    { text: r.rms.toFixed(2), options: { align: 'right', bold: r.rms === bestRms, color: r.rms === bestRms ? ORANGE : INK } },
    { text: r.pout.toFixed(1), options: { align: 'right', bold: r.pout === bestOut, color: r.pout === bestOut ? ORANGE : INK } },
    { text: r.ach_rms.toFixed(2), options: { align: 'right', bold: r.ach_rms === bestAch, color: r.ach_rms === bestAch ? ORANGE : INK } },
    { text: r.drive_trace.toPrecision(3), options: { align: 'right' } },
    { text: (r.level >= 0 ? '+' : '') + r.level.toFixed(2), options: { align: 'right' } },
  ]);
});
s.addTable(t5, {
  x: M, y: 1.66, w: W - 2 * M, fontFace: BODY, fontSize: 10.5, color: INK,
  border: { type: 'solid', color: 'DDE1E3', pt: 0.5 },
  colW: [0.72, 3.02, 0.72, 0.62, 1.02, 0.92, 1.12, 1.32, 1.02], rowH: 0.295,
});
caption(s, 'rms dB and % out score against the flat specification. vs floor scores against the best achievable '
  + 'response — the law’s own contribution. Orange marks the best value in each column.', 5.9);
s.addNotes(
  'Two acceptance criteria and they now mostly agree. congruence leads on rms (5.23), '
  + 'opt diag fast on lines-out (10.7 %), opt diagonal on error-against-floor (3.99, '
  + 'with congruence at 3.48 actually lowest — check the bold).\n\n'
  + 'Key rows to point at: 09 versus 10 (pseudoinverse, cap on versus off — 10.47 to '
  + '7.01 dB) and 11 versus 12 (buzz, 7.90 to 5.91). Those are slide 6. And run 12 at '
  + '5.91 dB sits 0.68 dB behind the best run in the whole matrix despite having no '
  + 'error feedback at all — that is slide 8.\n\n'
  + 'Caveat on run 01: its optimal single rescale is +3.37 dB, which suggests it was '
  + 'saved short of convergence. It is the weakest number in the table.');

/* ------------------------------------------------- 6. the cap inverts */
s = pres.addSlide();
titleSlide(s, 6, 'The coherence cap inverts by family',
  'Same parameter, opposite meaning for feedback and open-loop laws');
fitImage(s, 'figs/cap_effect.png', 2.35, { x: M, y: 1.56, w: W - 2 * M, h: 3.22 });
statTile(s, M, 5.16, 3.3, '−3.47 dB', 'pseudoinverse_control\ngains from cap off', ORANGE);
statTile(s, M + 3.5, 5.16, 3.3, '−1.99 dB', 'buzz_control\ngains from cap off', ORANGE);
statTile(s, M + 7.0, 5.16, 3.3, '≤ 0.08 dB', 'three of four feedback\nlaws move by this', BLUE);
s.addNotes(
  'The operational consequence: for pseudoinverse_control and buzz_control, cap off '
  + '(1.0) is the CORRECT setting, not the reckless one. This is the reverse of the '
  + 'feedback-law intuition and nothing in the parameter name or the docstring warns '
  + 'of it.\n\n'
  + 'The level offsets tell the same story more starkly than rms: run 09 sat +13.55 dB '
  + 'above spec with the cap on and −1.28 with it off; run 11 went +9.38 to −0.30.\n\n'
  + 'match_trace is the one feedback law that loses a full 1.01 dB to the cap — but '
  + 'removing it costs 50x the drive power (0.12 to 6.02 V^2), so cap on is still the '
  + 'right call there.');

/* --------------------------------------------------------- 7. mechanism */
s = pres.addSlide();
titleSlide(s, 7, 'Why the cap breaks the open-loop laws',
  'The drive correlation is not incidental — it is the solution');
const steps = [
  ['1', 'The open-loop solve is nearly rank one',
    'Eigenvalue participation ratio 1.0–1.1 out of 6. The six shakers act as essentially one source.'],
  ['2', 'The responses cancel by precise inter-drive phasing',
    'That phasing is carried entirely in the off-diagonal terms of the drive CPSD.'],
  ['3', 'The cap shrinks off-diagonals, preserves the diagonal',
    'Drive power barely changes — but the cancellation is destroyed and responses add incoherently.'],
  ['4', 'A feedback loop re-converges around it. An open-loop law cannot.',
    'Hence +13.55 dB on run 09. The residual is shape error, not level: the best single rescale recovers only 0.30 dB.'],
];
let y = 1.72;
steps.forEach(([n, head, sub]) => {
  s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.04, w: 0.44, h: 0.44, fill: { color: ORANGE } });
  s.addText(n, { x: M, y: y + 0.04, w: 0.44, h: 0.44, isTextBox: true,
    fontFace: BODY, fontSize: 14, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
  s.addText(head, { x: M + 0.66, y: y, w: W - 2 * M - 0.66, h: 0.34, isTextBox: true,
    fontFace: BODY, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText(sub, { x: M + 0.66, y: y + 0.36, w: W - 2 * M - 0.66, h: 0.44, isTextBox: true,
    fontFace: BODY, fontSize: 13, color: INK2, margin: 0 });
  y += 1.06;
});
s.addShape(pres.ShapeType.roundRect, { x: M, y: 6.12, w: W - 2 * M, h: 0.74, fill: { color: PANEL }, rectRadius: 0.06 });
s.addText([
  { text: 'Confirmed in reverse: ', options: { bold: true, color: INK } },
  { text: 'the best achievable drive sits above 0.95 coherence on 49 % of its pairs. '
        + 'The cap constrains the solution away from the optimum by construction.',
    options: { color: INK2 } },
], { x: M + 0.2, y: 6.24, w: W - 2 * M - 0.4, h: 0.5, isTextBox: true, fontFace: BODY, fontSize: 13, margin: 0 });
s.addNotes(
  '_cap_drive_coherence shrinks each off-diagonal term whose coherence exceeds the cap, '
  + 'preserving every diagonal value, then re-projects onto the PSD cone by clipping '
  + 'negative eigenvalues. Because the diagonal is preserved the drive trace is nearly '
  + 'unchanged — 14.54 V^2 either way in the offline reconstruction — yet the predicted '
  + 'response level jumps 21 dB.\n\n'
  + 'The "shape not level" point is worth making: the optimal single rescale of run 09\'s '
  + 'entire drive improves it only from 10.47 to 10.17 dB. No gain correction rescues it.');

/* --------------------------------------- 8. feedback buys less than expected */
s = pres.addSlide();
titleSlide(s, 8, 'Feedback buys less than expected',
  'A law with no error feedback finishes inside the feedback cluster');
fitImage(s, 'figs/rms_vs_pout.png', 1.32, { x: M, y: 1.6, w: 5.6, h: 4.5 });
s.addText([
  { text: 'buzz_control has no feedback at all\n', options: { bold: true, fontSize: 17, color: INK, breakLine: true } },
  { text: 'last_response_cpsd appears in its signature and docstring and never in the body. '
        + 'last_output_cpsd is read only as a first-call sentinel. It recomputes the same '
        + 'open-loop solve every cycle.\n\n', options: { fontSize: 13.5, color: INK2, breakLine: true } },
  { text: 'Its one adaptive channel is the plant model\n', options: { bold: true, fontSize: 17, color: INK, breakLine: true } },
  { text: 'The FRF is re-identified during control, but sysid_response_cpsd — the source of '
        + 'the coherence and phase it matches — is frozen at system ID. It adapts to a '
        + 'changing plant, not to its own error.\n\n', options: { fontSize: 13.5, color: INK2, breakLine: true } },
  { text: 'So what separates the laws?\n', options: { bold: true, fontSize: 17, color: INK, breakLine: true } },
  { text: 'Not loop design. How sensibly each one picks a target inside the reachable set. '
        + 'buzz substitutes measured coherence and phase for the spec’s unreachable diagonal '
        + '— worth 1.1 dB and a third of the drive against pseudoinverse.',
    options: { fontSize: 13.5, color: INK2 } },
], { x: M + 6.0, y: 1.66, w: W - 2 * M - 6.0, h: 4.4, isTextBox: true,
     fontFace: BODY, margin: 0, valign: 'top' });
s.addNotes(
  'buzz_control at 5.91 dB / 13.6 % finishes 0.68 dB behind the best run in the matrix '
  + 'and ahead of both match_trace runs — with no error feedback whatsoever.\n\n'
  + 'The reading this supports: on this plant the 5.2-5.9 dB cluster is set by '
  + 'reachability and identification quality, not by loop design.\n\n'
  + 'Cheap experiment worth running: buzz_control with update_tf_during_control = N. '
  + 'That makes the law fully static after cycle one, so the difference against run 12 '
  + 'measures exactly what plant-model adaptation is worth with nothing else varying.');

/* ----------------------------------------- 9. law error vs plant limit */
s = pres.addSlide();
titleSlide(s, 9, 'Separating the law’s error from the plant’s limit',
  'Two independent scorings of the same run — they do not subtract');
fitImage(s, 'figs/error_vs_floor.png', 2.30, { x: M, y: 1.6, w: W - 2 * M, h: 3.4 });
statTile(s, M, 5.34, 3.3, '3.48 dB', 'congruence — lowest\nerror against the floor', ORANGE);
statTile(s, M + 3.5, 5.34, 3.3, '3.99 dB', 'the whole optimal-\ndiagonal family', BLUE);
statTile(s, M + 7.0, 5.34, 3.3, '3.57–3.92', 'the floor itself\n(dB rms, per run)');
s.addNotes(
  'The bar is error against what this plant can actually deliver — the law\'s own '
  + 'contribution. The diamond is total error against the flat specification. These are '
  + 'separate rms figures and do NOT subtract; an earlier analysis in this campaign '
  + 'treated their difference as the reachability limit and was wrong.\n\n'
  + 'Ranked against the floor the ordering changes from the flat-spec ordering: '
  + 'congruence 3.48-3.51, optimal-diagonal family a flat 3.99, buzz 4.20, pseudoinverse '
  + '5.48, match_trace 5.78. congruence sits +2.8 dB high in level, which costs it '
  + 'against the specification but not against what was reachable.\n\n'
  + 'The floor itself: for each line, solve over every X >= 0 for the drive whose '
  + 'predicted response diagonal comes closest to the spec, then take the rms of the '
  + 'per-line per-channel dB error over 901 lines x 8 channels.');

/* ---------------------------------------------------- 10. per-channel */
s = pres.addSlide();
titleSlide(s, 10, 'The same two channels dominate every law',
  '13X+ and 14X+ are the reachability-limited pair');
fitImage(s, 'figs/per_channel_mean.png', 1.99, { x: M, y: 1.6, w: 7.5, h: 4.3 });
s.addShape(pres.ShapeType.roundRect, { x: M + 7.9, y: 1.66, w: W - 2 * M - 7.9, h: 1.72, fill: { color: PANEL }, rectRadius: 0.06 });
s.addShape(pres.ShapeType.roundRect, { x: M + 7.9, y: 3.62, w: W - 2 * M - 7.9, h: 1.86, fill: { color: PANEL }, rectRadius: 0.06 });
s.addText([
  { text: 'Five channels hold\n', options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: '8X+ through 12X+ stay within 2.1 dB under every law. 7X+ is marginal — inside '
        + '2 dB for four of five laws, −3.6 dB under pseudoinverse.',
    options: { fontSize: 12.5, color: INK2 } },
], { x: M + 8.08, y: 1.8, w: W - 2 * M - 8.26, h: 1.86, isTextBox: true, fontFace: BODY, margin: 0, valign: 'top' });
s.addText([
  { text: 'Every law undershoots the pair\n', options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: 'congruence      −2.2 / −2.8\nbuzz            −5.3 / −6.0\n'
        + 'match_trace     −6.6 / −7.3\npseudoinverse   −8.5 / −9.2',
    options: { fontSize: 12, color: INK2, fontFace: 'Courier New' } },
], { x: M + 8.08, y: 3.78, w: W - 2 * M - 8.26, h: 1.64, isTextBox: true, fontFace: BODY, margin: 0, valign: 'top' });
s.addNotes(
  'These are the two directions six shakers cannot independently command. The laws '
  + 'differ mainly in how much drive they are willing to waste trying — congruence gives '
  + 'up soonest and scores best; pseudoinverse spends the most and does worst.\n\n'
  + 'The floor itself has the same shape: its per-channel means are +0.22, +2.14, +0.39, '
  + '+0.37, +0.24 and +0.39 dB on 7X+ through 12X+, and -1.79 / -2.20 on 13X+ / 14X+. '
  + 'So three channels carry the floor — 8X+ overshooting and this pair undershooting.');

/* --------------------------------------------------------- 11. cross-terms */
s = pres.addSlide();
titleSlide(s, 11, 'Cross-terms: the specification asked for zero',
  'Nothing in any of these laws controls the off-diagonals');
fitImage(s, 'figs/cross_terms.png', 2.47, { x: M, y: 1.54, w: W - 2 * M, h: 3.22 });
statTile(s, M, 5.12, 3.3, '0.25 – 0.89', 'median response\ncoherence |γ|', ORANGE);
statTile(s, M + 3.5, 5.12, 3.3, '1.15 – 1.5', 'effective response\ndirections, out of 8');
statTile(s, M + 7.0, 5.12, 3.3, '0.967', 'coherence of the BEST\nachievable response', ORANGE);
caption(s, 'Rank 5 < 8 control channels, so a diagonal response CPSD is unreachable — and the optimum is more correlated than almost every law delivered.', 6.74);
s.addNotes(
  'The flat spec is diagonal — every off-diagonal term zero. Every one of these laws '
  + 'matches the DIAGONAL and lets the cross-structure fall out. It does not fall out '
  + 'small: off-diagonal magnitudes sit within a few dB of the 1e-3 diagonal.\n\n'
  + 'Zero cross-terms is unreachable, not merely hard: H X H^H has rank at most 5 in an '
  + '8-dimensional space, while a genuinely diagonal 8x8 CPSD has rank 8. This is a '
  + 'second, independent way the spec sits outside the achievable set.\n\n'
  + 'Run 10 is the control experiment: the only run with substantially independent '
  + 'responses (median 0.245, participation 4.46), bought with 5.38 V^2 — seventy times '
  + 'the optimal-diagonal family — to finish 1.4 dB worse. Decorrelating costs a lot and '
  + 'buys nothing, because nothing in the acceptance criterion looks at it.\n\n'
  + 'If cross-terms matter for a real article, none of these laws gives control over '
  + 'them and no parameter setting changes that.');

/* --------------------------------------------------------- 12. drive power */
s = pres.addSlide();
titleSlide(s, 12, 'Drive power spans a factor of 128',
  'If drive headroom binds rather than control accuracy, the ordering changes completely');
fitImage(s, 'figs/drive_power.png', 2.23, { x: M, y: 1.62, w: W - 2 * M, h: 3.5 });
statTile(s, M, 5.46, 3.3, '0.047 V²', 'congruence, cap on —\nthe cheapest run', BLUE);
statTile(s, M + 3.5, 5.46, 3.3, '1.47 V²', 'buzz, cap off — 20×\nfor 0.3 dB worse', ORANGE);
statTile(s, M + 7.0, 5.46, 3.3, '6.02 V²', 'match_trace, cap off —\n50× its own capped run', ORANGE);
s.addNotes(
  'The open-loop laws are expensive. buzz uncapped uses 1.47 V^2, about twenty times the '
  + 'optimal-diagonal family, for a 0.3 dB worse score.\n\n'
  + 'If drive headroom is the binding constraint rather than control accuracy, '
  + 'match_diagonal_congruence with the cap on wins outright at 0.047 V^2 — and that is '
  + 'the one case where keeping the cap on is clearly right, since it costs congruence '
  + 'only 0.08 dB.');

/* ------------------------------------------------------------- 13. caveats */
s = pres.addSlide();
titleSlide(s, 13, 'Caveats and open defects', 'What these numbers do not support');
const cav = [
  ['Runs 05 / 06 did not test what they were meant to',
   'optimal_diagonal_control_fast disables its Burer-Monteiro path whenever the FRF changes between calls. With update on, every solve fell back to the base SDP — they are repeats of 03 / 04.'],
  ['The cap barely applied in the optimal-diagonal laws',
   'It reaches only SDP-refined bins, capped at max_bins_per_update = 20 across 901 lines. Their null result may mean “barely applied”, not “does not matter”.'],
  ['pseudoinverse_control has no startup guard',
   'The only law in the set without the first-cycle clamp. On frame one the FRF is a single unaveraged estimate and the command can be decades high.'],
  ['Identification scatter is large',
   'Median cond(H) 2,447–25,928 across twelve identifications of one deterministic plant. Differences below a few tenths of a dB are not real.'],
  ['One level, one averaging setting, one plant, no repeats',
   'No per-law variance estimate exists. The plant is lightly damped, which is the harder case.'],
];
y = 1.72;
cav.forEach(([head, sub]) => {
  s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.06, w: 0.2, h: 0.2, fill: { color: ORANGE } });
  s.addText(head, { x: M + 0.42, y: y, w: W - 2 * M - 0.42, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText(sub, { x: M + 0.42, y: y + 0.33, w: W - 2 * M - 0.42, h: 0.58, isTextBox: true,
    fontFace: BODY, fontSize: 12, color: INK2, margin: 0 });
  y += 1.02;
});
s.addNotes(
  'Two more worth mentioning if asked. Run 01 may have been saved short of convergence — '
  + 'its optimal rescale is +3.37 dB, the only large one-sided offset under an integrating '
  + 'law. And an offline predictor used during the campaign called every ordering '
  + 'correctly and no magnitude within 2 dB, because it freezes the FRF the live laws '
  + 'keep updating.\n\n'
  + 'Also: the profiles carry no warning or abort levels, so the +/-6 dB bands are '
  + 'synthesized from the specification. Putting real tolerances in the profile would make '
  + 'these numbers contractual rather than nominal.');

/* ---------------------------------------------------------------- 14. next */
s = pres.addSlide();
s.background = { color: DARK };
s.addText('Next', { x: M, y: 0.62, w: W - 2 * M, h: 0.7, isTextBox: true,
  fontFace: HEAD, fontSize: 34, bold: true, color: 'FFFFFF', margin: 0 });
const next = [
  ['Find out why the plant is rank 5', 'If the sixth shaker is redundant by construction it matters for the rig, not just the model.'],
  ['Higher damping, same geometry', 'higherz has identical mass and stiffness, ζ 0.83 → 3.34 %. Floor unchanged (4.05 vs 4.09) — it isolates identification quality.'],
  ['buzz_control with the FRF update off', 'Makes the law fully static after cycle one. One run measures what plant-model adaptation is worth.'],
  ['Raise max_bins_per_update, repeat 03 / 04', 'Settles whether the optimal-diagonal cap null result is real.'],
  ['Decide whether cross-terms are in scope', 'Currently a diagonal-only comparison by default rather than by decision.'],
  ['Add a startup level cap to the remaining laws', 'Open defect on both optimal-diagonal laws and pseudoinverse_control.'],
];
y = 1.6;
next.forEach(([head, sub], i) => {
  s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.02, w: 0.42, h: 0.42, fill: { color: i < 2 ? ORANGE : '5C6E7A' } });
  s.addText(String(i + 1), { x: M, y: y + 0.02, w: 0.42, h: 0.42, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: i < 2 ? '2B3740' : 'FFFFFF',
    align: 'center', valign: 'middle', margin: 0 });
  s.addText(head, { x: M + 0.62, y: y, w: 5.3, h: 0.46, isTextBox: true,
    fontFace: BODY, fontSize: 15, bold: true, color: 'FFFFFF', margin: 0, valign: 'middle' });
  s.addText(sub, { x: M + 6.1, y: y, w: W - M - 6.1 - M, h: 0.46, isTextBox: true,
    fontFace: BODY, fontSize: 12.5, color: 'B9C5CC', margin: 0, valign: 'middle' });
  y += 0.86;
});
s.addText('Report: Claude outputs/law_comparison_report_2026-09-15  ·  data: report_data.json, crossterm_data.json, singular_value_data.json',
  { x: M, y: 6.72, w: W - 2 * M, h: 0.34, isTextBox: true,
    fontFace: BODY, fontSize: 11, color: 'A4B4BE', margin: 0 });
s.addNotes(
  'Items 1 and 2 are the ones to argue about. The rank question is not a modelling '
  + 'curiosity: if the sixth shaker is redundant by construction then the rig has a drive '
  + 'channel doing nothing and the problem has always been 8x5.\n\n'
  + 'Set expectations correctly on the higher-damping campaign: it will NOT move the '
  + 'floor. It tests only whether the law spread narrows because identification improves '
  + '— conditioning over the live directions halves, 68 to 33.');

pres.writeFile({ fileName: 'law_comparison_deck.pptx' })
  .then(f => console.log('wrote', f));
