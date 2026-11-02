// Control-law reference deck. One slide per law: flow chart, algorithm,
// parameter string, measured result, known defects.
const pptxgen = require('pptxgenjs');

const INK = '36454F', INK2 = '5E6B73', MUTED = '8A959B';
const PANEL = 'F2F2F2', DARK = '2B3740';
const ORANGE = 'EB6834', BLUE = '2A78D6', GREEN = '1BAF7A', VIOLET = '4A3AA7';
const HEAD = 'Cambria', BODY = 'Calibri', MONO = 'Courier New';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'nhunter';
pres.title = 'Control laws in the Rattlesnake MIMO random environment';
const W = 13.3, M = 0.6;

function header(s, n, title, tag) {
  s.addText(title, { x: M, y: 0.32, w: 9.4, h: 0.62, isTextBox: true,
    fontFace: HEAD, fontSize: 27, bold: true, color: INK, margin: 0 });
  if (tag) {
    s.addShape(pres.ShapeType.roundRect, { x: M, y: 1.0, w: 4.1, h: 0.33,
      fill: { color: PANEL }, rectRadius: 0.05 });
    s.addText(tag, { x: M + 0.12, y: 1.0, w: 3.86, h: 0.33, isTextBox: true,
      fontFace: MONO, fontSize: 10.5, color: INK2, valign: 'middle', margin: 0 });
  }
  s.addShape(pres.ShapeType.ellipse, { x: W - M - 0.46, y: 0.36, w: 0.46, h: 0.46,
    fill: { color: PANEL } });
  s.addText(String(n), { x: W - M - 0.46, y: 0.36, w: 0.46, h: 0.46, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: INK2,
    align: 'center', valign: 'middle', margin: 0 });
}

// Landscape re-renders of the document's flow charts. The portrait originals
// are authored for a 560 px page column; scaled into a slide column their node
// text falls to ~4 px, so the deck uses LR variants at larger type.
function flow(s, name, ar, box) {
  let w = box.w, h = w / ar;
  if (h > box.h) { h = box.h; w = h * ar; }
  s.addImage({ path: `lr/${name}.png`,
    x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h });
}

function eqBox(s, x, y, w, h, lines) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h,
    fill: { color: PANEL }, rectRadius: 0.05 });
  const runs = [];
  lines.forEach((ln, i) => {
    const comment = ln.startsWith('#');
    runs.push({ text: comment ? ln.slice(1).trim() : ln,
      options: { fontFace: MONO, fontSize: comment ? 10 : 11.5,
        color: comment ? MUTED : INK, italic: comment,
        breakLine: i < lines.length - 1 } });
  });
  s.addText(runs, { x: x + 0.16, y: y + 0.12, w: w - 0.32, h: h - 0.24,
    isTextBox: true, margin: 0, valign: 'top', lineSpacingMultiple: 1.12 });
}

function factRow(s, x, y, w, items) {
  let yy = y;
  items.forEach(([k, v]) => {
    s.addText(k, { x, y: yy, w: 2.05, h: 0.28, isTextBox: true,
      fontFace: BODY, fontSize: 11.5, bold: true, color: INK, margin: 0 });
    s.addText(v, { x: x + 2.1, y: yy, w: w - 2.1, h: 0.28, isTextBox: true,
      fontFace: BODY, fontSize: 11.5, color: INK2, margin: 0 });
    yy += 0.3;
  });
  return yy;
}

function resultTable(s, x, y, w, rows) {
  const t = [[
    { text: 'Run', options: { bold: true, color: INK, fill: { color: PANEL } } },
    { text: 'Cap', options: { bold: true, color: INK, fill: { color: PANEL } } },
    { text: 'rms dB', options: { bold: true, color: INK, fill: { color: PANEL }, align: 'right' } },
    { text: '% out', options: { bold: true, color: INK, fill: { color: PANEL }, align: 'right' } },
    { text: 'vs floor', options: { bold: true, color: INK, fill: { color: PANEL }, align: 'right' } },
    { text: 'drive V²', options: { bold: true, color: INK, fill: { color: PANEL }, align: 'right' } },
  ]];
  rows.forEach(r => t.push([
    { text: r[0], options: { fontFace: MONO, fontSize: 10 } },
    { text: r[1] },
    { text: r[2], options: { align: 'right', bold: r[5] === true, color: r[5] === true ? ORANGE : INK } },
    { text: r[3], options: { align: 'right' } },
    { text: r[4], options: { align: 'right' } },
    { text: r[6], options: { align: 'right' } },
  ]));
  s.addTable(t, { x, y, w, fontFace: BODY, fontSize: 10.5, color: INK,
    border: { type: 'solid', color: 'DDE1E3', pt: 0.5 },
    colW: [w * 0.11, w * 0.12, w * 0.19, w * 0.17, w * 0.2, w * 0.21], rowH: 0.27 });
}

function note(s, x, y, w, h, title, text, accent) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h,
    fill: { color: PANEL }, rectRadius: 0.05 });
  s.addText([
    { text: title + '\n', options: { bold: true, fontSize: 12.5,
      color: accent || INK, breakLine: true } },
    { text: text, options: { fontSize: 11, color: INK2 } },
  ], { x: x + 0.16, y: y + 0.12, w: w - 0.32, h: h - 0.24, isTextBox: true,
       fontFace: BODY, margin: 0, valign: 'top' });
}

/* ------------------------------------------------------------- 1. title */
let s = pres.addSlide();
s.background = { color: DARK };
s.addText('Control laws in the Rattlesnake', { x: M, y: 1.95, w: W - 2 * M, h: 0.72,
  isTextBox: true, fontFace: HEAD, fontSize: 40, bold: true, color: 'FFFFFF', margin: 0 });
s.addText('MIMO random environment', { x: M, y: 2.72, w: W - 2 * M, h: 0.68,
  isTextBox: true, fontFace: HEAD, fontSize: 32, color: ORANGE, margin: 0 });
s.addText('Algorithms, equations and flow charts  ·  seven subjects  ·  6-12-8 system',
  { x: M, y: 3.72, w: W - 2 * M, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 16, color: 'D6DDE2', margin: 0 });
s.addText([
  { text: 'control_laws/', options: { fontFace: MONO, breakLine: true } },
  { text: '15 September 2026  ·  sections 5–7 added after runs 09–12' },
], { x: M, y: 5.6, w: W - 2 * M, h: 0.8, isTextBox: true,
     fontFace: BODY, fontSize: 12, color: '9FAEB8', margin: 0 });
s.addNotes(
  'This deck follows the reference document section for section. Six control '
  + 'laws plus the achievable-response predictor that supplies the denominator '
  + 'every comparison is measured against.\n\n'
  + 'Sections 5, 6 and 7 — pseudoinverse_control, buzz_control, and the '
  + 'coherence-cap behaviour — were added on 15 September after runs 09-12. The '
  + 'original document predated those runs and was materially incomplete: buzz '
  + 'in particular produced the most surprising result of the campaign.');

/* ------------------------------------------------ 2. common interface */
s = pres.addSlide();
header(s, 2, 'What every law receives', 'the control() contract');
eqBox(s, M, 1.55, 6.5, 4.0, [
  '# inputs, every cycle',
  'specification        G_xx   (F,M,M)',
  'warning / abort      levels',
  'transfer_function    H      (F,M,N)',
  'noise_response_cpsd         (F,M,M)',
  'sysid_response_cpsd         (F,M,M)',
  'multiple_coherence          (F,M)',
  'frames, total_frames',
  'extra_parameters     str',
  'last_response_cpsd   G_yy   (F,M,M)',
  'last_output_cpsd     X_prev (F,N,N)',
  '',
  '# output',
  'drive CPSD           X      (F,N,N)',
  '',
  '# the one hard requirement',
  'X(f) must be Hermitian positive',
  'semidefinite at every line f.',
]);
factRow(s, M + 6.9, 1.62, W - M - (M + 6.9), [
  ['M, N', '8 control channels, 6 drives'],
  ['F', '901 in-band lines, 100–1000 Hz'],
  ['loaded by', 'importlib.util.spec_from_file_location'],
  ['law types', '1 = generator, 2 = class, else function'],
]);
note(s, M + 6.9, 3.15, W - M - (M + 6.9), 1.35,
  'X ⪰ 0 is what makes the laws differ',
  'A drive CPSD that is not positive semidefinite is not physically realisable. '
  + 'How each law guarantees it — by scalar, by cone constraint, by factoring, '
  + 'by congruence — is the single design choice that determines both its power '
  + 'and its limits.', ORANGE);
note(s, M + 6.9, 4.68, W - M - (M + 6.9), 1.55,
  'Two laws never read last_response_cpsd',
  'pseudoinverse_control and buzz_control accept it and ignore it. They have no '
  + 'error term of any kind — they recompute an open-loop solve every cycle and '
  + 'adapt only through the FRF. Sections 5 and 6.', VIOLET);
s.addNotes(
  'Every law is handed the same eleven arguments and returns one thing: the '
  + 'drive CPSD. The only hard requirement is that X be Hermitian positive '
  + 'semidefinite at every frequency line — otherwise the commanded drive is not '
  + 'physically realisable.\n\n'
  + 'Rattlesnake loads laws by file path with importlib, so there is no package '
  + 'context and no relative imports are possible inside a law file.');

/* ----------------------------------------------------- 3. the family */
s = pres.addSlide();
header(s, 3, 'Six laws, one problem, six different guarantees');
const fam = [
  ['match_trace_pseudoinverse', 'uniform positive scalar', '1', 'feedback', '8.22'],
  ['optimal_diagonal_control', 'explicit SDP cone constraint', 'N² (full X)', 'feedback', '5.62'],
  ['optimal_diagonal_control_fast', 'structural, X = L Lᴴ', '2Nr', 'feedback', '5.64'],
  ['match_diagonal_congruence', 'structural, D X D', 'N', 'feedback', '5.31'],
  ['pseudoinverse_control', 'automatic — congruence of G_xx', '0 (open loop)', 'open', '10.47'],
  ['buzz_control', 'automatic — congruence of S′', '0 (open loop)', 'open', '7.90'],
];
const t3 = [[
  { text: 'Law', options: { bold: true, color: INK, fill: { color: PANEL } } },
  { text: 'How X ⪰ 0 is guaranteed', options: { bold: true, color: INK, fill: { color: PANEL } } },
  { text: 'DOF per line', options: { bold: true, color: INK, fill: { color: PANEL } } },
  { text: 'Loop', options: { bold: true, color: INK, fill: { color: PANEL } } },
  { text: 'rms dB, cap on', options: { bold: true, color: INK, fill: { color: PANEL }, align: 'right' } },
]];
fam.forEach(r => t3.push([
  { text: r[0], options: { fontFace: MONO, fontSize: 10.5 } },
  { text: r[1] },
  { text: r[2] },
  { text: r[3], options: { color: r[3] === 'open' ? ORANGE : BLUE, bold: true } },
  { text: r[4], options: { align: 'right' } },
]));
s.addTable(t3, { x: M, y: 1.5, w: W - 2 * M, fontFace: BODY, fontSize: 11.5, color: INK,
  border: { type: 'solid', color: 'DDE1E3', pt: 0.5 },
  colW: [3.5, 3.6, 1.9, 1.2, 1.9], rowH: 0.36 });
note(s, M, 4.6, (W - 2 * M - 0.4) / 2, 1.9,
  'The guarantees buy shaping, not validity',
  'The two open-loop laws need no machinery at all: X = H⁺ S (H⁺)ᴴ is a '
  + 'congruence transform of a PSD matrix, so validity is automatic. Everything '
  + 'the other four add buys the ability to shape the drive — and on this plant '
  + 'that shaping is worth between 0 and 1.1 dB against a well-chosen target.', ORANGE);
note(s, M + (W - 2 * M - 0.4) / 2 + 0.4, 4.6, (W - 2 * M - 0.4) / 2, 1.9,
  'Cap-on understates the open-loop pair',
  'The rms column is the cap-on run for every law, which is the right setting '
  + 'for the four feedback laws and the wrong one for the other two. Uncapped, '
  + 'pseudoinverse measures 7.01 dB and buzz 5.91 — see section 7.', VIOLET);
s.addNotes(
  'This is the spine of the whole document. Each law solves the same problem — '
  + 'choose a drive CPSD whose predicted response matches the specification — '
  + 'and what separates them is how they keep X positive semidefinite while '
  + 'doing it.\n\n'
  + 'Degrees of freedom per line is the useful axis: match_trace has exactly one '
  + '(a scalar), congruence has N, the optimal-diagonal laws have the full '
  + 'matrix, and the open-loop pair has none because they do not iterate at all.');

/* ---------------------------------------------- 4-9. one slide per law */
const LAWS = [
  { n: 4, name: 'match_trace_pseudoinverse', tag: 'function · control_laws.py',
    flow: '01_match_trace', ar: 6.817,
    blurb: 'The baseline, and the only law with a long validation history. One '
      + 'open-loop solve at startup, then a single real scalar per line.',
    eq: ['# first call — open loop',
      'H⁺ = pinv(H, rcond)',
      'X  = H⁺ G_xx (H⁺)ᴴ',
      's  = min(1, 10^(startup_cap_db/10)·tr(G_xx)/tr(X))',
      'X ← s·X',
      '',
      '# every call after — closed loop on total power',
      'c(f) = tr(G_xx(f)) / tr(G_yy(f))',
      'X(f) ← c(f) · X_prev(f)'],
    params: 'rcond, max_drive_coherence, startup_test_level_cap_db\ndefaults 1e-15, 1.0, −9.0',
    rows: [['01', '0.95', '8.22', '27.5', '7.16', false, '0.120'],
           ['02', 'off', '7.21', '21.1', '5.78', true, '6.019']],
    noteT: 'Startup cap ≠ coherence cap',
    noteB: 'startup_test_level_cap_db is a level clamp on cycle one; '
      + 'max_drive_coherence is the coherence cap. Unrelated parameters.',
    noteC: BLUE,
    notes: 'Why it is safe: a uniform positive real scalar can never turn a '
      + 'valid drive CPSD into an invalid one, and after startup the FRF is '
      + 'never inverted again. Why it is limited: if channel 3 is 5 dB high and '
      + 'channel 5 is 5 dB low, the trace is correct and the law does nothing.\n\n'
      + 'Run 01 may have been saved short of convergence — its optimal single '
      + 'rescale is +3.37 dB, which would improve it to 7.50 dB. It is the '
      + 'weakest number in the comparison.' },

  { n: 5, name: 'optimal_diagonal_control', tag: 'class · optimal_diagonal_control.py',
    flow: '02_optimal_diagonal', ar: 9.679,
    blurb: 'Solves a semidefinite program per frequency bin for the full drive '
      + 'matrix, with the PSD cone as an explicit constraint.',
    eq: ['# per bin, subject to X ⪰ 0',
      'min  Σ_m ( 10·log10( [H X Hᴴ]_mm / y_m ) )²',
      '',
      '# plus an explicit coherence constraint',
      '|X_ij| ≤ c · sqrt(X_ii · X_jj)',
      '',
      '# scheduled: only max_bins_per_update bins',
      '# are SDP-refined per call; the rest carry',
      '# the buzz baseline forward'],
    params: 'rcond, ridge, max_bins_per_update, gain, max_drive_coherence\nused here 1e-6, 0.05, 20, 1.0, 0.95',
    rows: [['03', '0.95', '5.62', '11.2', '3.99', true, '0.0747'],
           ['04', 'off', '5.62', '11.2', '3.99', false, '0.0790']],
    noteT: 'Open defect: no startup level cap',
    noteB: 'Neither optimal-diagonal law takes startup_test_level_cap_db — it '
      + 'commands the full specification immediately.',
    noteC: ORANGE,
    notes: 'Best of the set on lines-out (11.2 %) and essentially tied with '
      + 'congruence on rms. The cap made no measurable difference — but see the '
      + 'caveat: the cap reaches only SDP-refined bins, capped at 20 per call '
      + 'across 901 lines, so most bins carried the deliberately uncapped buzz '
      + 'baseline in both runs. That null result may mean "barely applied" '
      + 'rather than "does not matter".' },

  { n: 6, name: 'optimal_diagonal_control_fast', tag: 'subclass · optimal_diagonal_control_fast.py',
    flow: '03_optimal_diagonal_fast', ar: 3.532,
    blurb: 'Same objective, but X = L Lᴴ makes positive semidefiniteness '
      + 'structural instead of constrained — no cone projection needed.',
    eq: ['# Burer-Monteiro factoring',
      'X = L Lᴴ,  L ∈ ℂ^(N×r)',
      '',
      '# X ⪰ 0 for ANY L — the constraint',
      '# disappears from the problem entirely',
      '',
      '# spurious-minimum threshold',
      'r ≳ sqrt(2M) = 4  for M = 8',
      '# bm_rank was set to 6'],
    params: 'rcond, ridge, max_bins_per_update, gain, max_drive_coherence, bm_rank\nused here 1e-6, 0.05, 20, 1.0, 0.95, 6',
    rows: [['05', '0.95', '5.64', '10.9', '3.99', false, '0.0760'],
           ['06', 'off', '5.66', '10.7', '3.99', true, '0.0756']],
    noteT: 'These runs did not test what they were meant to',
    noteB: 'The fast path disables itself when the FRF changes between calls, so '
      + 'runs 05 and 06 are repeats of 03 and 04.',
    noteC: ORANGE,
    notes: 'Their close agreement with runs 03 and 04 is NOT evidence that the '
      + 'factored solve matches the SDP. No timing difference was observed at '
      + 'the rig, which is consistent with the fast path never engaging.\n\n'
      + 'To actually test the claim, rerun with update_tf_during_control = N.' },

  { n: 7, name: 'match_diagonal_congruence', tag: 'class · diagonal_congruence_control.py',
    flow: '04_congruence', ar: 3.769,
    blurb: 'One real gain per drive per line, applied as a congruence. Keeps '
      + 'validity structural like the fast law, but with N degrees of freedom '
      + 'instead of 2Nr.',
    eq: ['# measured log error',
      'ε_m = log(target_m / achieved_m)',
      '',
      '# sensitivity — every row sums to exactly 2',
      'S_mi = ∂log(y_m)/∂u_i',
      '     = 2·Re(H_mi (XHᴴ)_im) / y_m',
      '',
      '# regularised Gauss-Newton, then congruence',
      'u = argmin ‖S u − ε‖² + λ‖u‖²',
      'D = diag(exp u);   X ← D X D'],
    params: 'rcond, max_drive_coherence, startup_cap_db, ridge, iters, gain, max_step_db, ceiling\nused here 1e-3, 0.95, −9, 0.5, 3, 0.1, 0.5, 12',
    rows: [['07', '0.95', '5.31', '19.8', '3.51', false, '0.0469'],
           ['08', 'off', '5.23', '18.7', '3.48', true, '0.2107']],
    noteT: 'Validity is structural for ANY real D',
    noteB: 'X ⪰ 0 ⟹ D X D ⪰ 0 for any real D. u_i = ε/2 for all i recovers '
      + 'match_trace exactly.',
    noteC: GREEN,
    notes: 'Best rms in the twelve-run matrix (5.23) and lowest error against '
      + 'the achievable floor (3.48). Also the cheapest run in the set on drive '
      + 'power with the cap on: 0.047 V².\n\n'
      + 'Its weakness is level: it sits +2.8 dB high, which costs it against the '
      + 'specification but not against what was reachable. That is why it leads '
      + 'the floor-relative ranking more clearly than the flat-spec one.' },

  { n: 8, name: 'pseudoinverse_control', tag: 'function · control_laws.py',
    flow: '06_pseudoinverse', ar: 3.98,
    blurb: 'The simplest and oldest law: one open-loop solve against the raw '
      + 'specification, recomputed identically every cycle. No feedback, no '
      + 'trace management, no startup guard.',
    eq: ['# every call, identically',
      'H⁺ = pinv(H, rcond)',
      'X  = H⁺ G_xx (H⁺)ᴴ',
      'X ← cap(X, max_drive_coherence)',
      '',
      '# that is the entire law',
      '# last_response_cpsd is accepted',
      '# and never read'],
    params: 'rcond, max_drive_coherence\ndefaults 1e-15, 1.0   — no third parameter',
    rows: [['09', '0.95', '10.47', '42.1', '9.55', false, '5.449'],
           ['10', 'off', '7.01', '17.8', '5.48', true, '5.377']],
    noteT: 'Open defect: no startup guard at all',
    noteB: 'The only law without the first-cycle clamp. On frame one the command '
      + 'can be decades high.',
    noteC: ORANGE,
    notes: 'Why it is expensive: the flat specification is diagonal and hence '
      + 'rank 8, while H X Hᴴ has rank at most 5 on this plant. Inverting a '
      + 'maximally unreachable target with no trace or coherence management '
      + 'burns drive power trying to manufacture response components that do '
      + 'not exist in the column space — 5.38 V², seventy times the '
      + 'optimal-diagonal family, for a worse score.\n\n'
      + 'Worst in the matrix on both criteria with the cap on. The cap costs it '
      + '3.47 dB; section 7 explains why.' },

  { n: 9, name: 'buzz_control', tag: 'function · control_laws.py',
    flow: '07_buzz', ar: 5.483,
    blurb: 'pseudoinverse_control with one substitution — and that substitution '
      + 'is worth more than every feedback mechanism in the rest of this deck.',
    eq: ['# replace the spec coherence and phase',
      '# with the MEASURED ones from the buzz test',
      'γ²_mn = |G_buzz,mn|² /(G_buzz,mm G_buzz,nn)',
      'φ_mn  = ∠G_buzz,mn',
      'a_m   = G_xx,mm          # spec autospectra',
      "S'_mn = e^(iφ_mn)·sqrt(γ²_mn · a_m · a_n)",
      '',
      '# then the same open-loop solve',
      "X = H⁺ S' (H⁺)ᴴ"],
    params: 'rcond, max_drive_coherence, startup_test_level_cap_db\ndefaults 1e-15, 1.0, −9.0  (shares match_trace’s parser)',
    rows: [['11', '0.95', '7.90', '30.2', '6.75', false, '0.981'],
           ['12', 'off', '5.91', '13.6', '4.20', true, '1.473']],
    noteT: 'It inverts a target the plant can actually produce',
    noteB: 'The spec asks for zero coherence, which a rank-5 plant cannot give. '
      + 'Same solver, reachable target: 1.1 dB better.',
    noteC: GREEN,
    notes: 'Uncapped, buzz finishes 0.68 dB behind the best run in the entire '
      + 'twelve-run matrix and ahead of both match_trace runs — with no error '
      + 'feedback whatsoever.\n\n'
      + 'Second limitation worth knowing: sysid_response_cpsd is assigned only '
      + 'during system ID and is never refreshed during control, so the target '
      + 'shape is frozen for the whole run. Only the FRF adapts.\n\n'
      + 'The generator and class forms are NOT equivalent: neither has the cap, '
      + 'neither takes rcond, neither has the startup guard, and '
      + 'buzz_control_class.control() ignores last_response_cpsd entirely.' },
];

const COL = 6.35, RX = M + COL + 0.4, RW = W - M - RX;
LAWS.forEach(L => {
  const sl = pres.addSlide();
  header(sl, L.n, L.name, L.tag);
  sl.addText(L.blurb, { x: M, y: 1.44, w: W - 2 * M, h: 0.42, isTextBox: true,
    fontFace: BODY, fontSize: 13, color: INK2, margin: 0 });
  flow(sl, L.flow, L.ar, { x: M, y: 1.98, w: W - 2 * M, h: 2.45 });
  eqBox(sl, M, 4.58, COL, 2.22, L.eq);
  sl.addText('Parameter string', { x: RX, y: 4.58, w: RW, h: 0.24, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0 });
  sl.addText(L.params, { x: RX, y: 4.82, w: RW, h: 0.5, isTextBox: true,
    fontFace: MONO, fontSize: 10, color: INK2, margin: 0 });
  resultTable(sl, RX, 5.22, RW, L.rows);
  note(sl, RX, 6.14, RW, 0.72, L.noteT, L.noteB, L.noteC);
  sl.addNotes(L.notes);
});

/* ------------------------------------------------- 10. achievable_response */
s = pres.addSlide();
header(s, 10, 'achievable_response', 'not a control law · achievable_response.py');
s.addText('The denominator every comparison is measured against: for each line, '
  + 'the best response any physically valid drive could produce.',
  { x: M, y: 3.16, w: W - 2 * M, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 13, color: INK2, margin: 0 });
eqBox(s, M, 3.64, 6.35, 2.4, [
  '# per line, over every valid drive',
  'min over X ⪰ 0 of',
  '  sqrt( mean_m [10·log10(a_m / y_m)]² )',
  'where a = diag(H X Hᴴ)',
  '',
  '# X = L Lᴴ makes X ⪰ 0 structural;',
  '# Levenberg-Marquardt, multi-restart',
  '# restrict_rcond limits L to the drive',
  '# directions surviving that cutoff']);
note(s, RX, 3.64, RW, 1.62, 'Two floors, and they agree',
  'At rcond 1e-3 — what the laws actually invert with — the floor is '
  + '3.57–3.92 dB rms. Unrestricted but over the plant\'s real rank-5 subspace '
  + 'it is 3.56–3.90. rcond is not costing headroom; it is rejecting a '
  + 'direction that does not exist.', GREEN);
note(s, RX, 5.42, RW, 1.06, 'These are two independent scorings',
  'A run\'s error against the floor and its error against the specification do '
  + 'NOT subtract. Both appear in the comparison report for that reason.', ORANGE);
flow(s, '05_achievable', 5.026, { x: M, y: 1.46, w: W - 2 * M, h: 1.6 });
s.addText('per-channel floor, dB', { x: RX, y: 6.34, w: RW, h: 0.24,
  isTextBox: true, fontFace: BODY, fontSize: 11.5, bold: true, color: INK, margin: 0 });
s.addText('7X+ +0.22   8X+ +2.14   9X+ +0.39   10X+ +0.37\n'
  + '11X+ +0.24  12X+ +0.39  13X+ −1.79  14X+ −2.20',
  { x: RX, y: 6.58, w: RW, h: 0.42, isTextBox: true,
    fontFace: MONO, fontSize: 9.5, color: INK2, margin: 0 });
s.addNotes(
  'Not a control law — a predictor. It answers "how well could ANY law have '
  + 'done here", which turns a bare rms number into a meaningful one.\n\n'
  + 'Five channels sit inside half a dB of the specification; three carry the '
  + 'whole floor — 8X+ overshooting and the 13X+/14X+ pair undershooting. Those '
  + 'last two are the directions six shakers cannot independently command.\n\n'
  + 'Caution: the pooled rms (3.9) is much larger than the rms of those '
  + 'per-channel means (1.29) because per-line scatter is worse than the '
  + 'channel averages suggest.');

/* ----------------------------------------------------- 11. the cap */
s = pres.addSlide();
header(s, 11, 'The cap inverts by family');
const capRows = [
  ['match_diagonal_congruence', 'feedback', '5.31', '5.23', '−0.08', '+2.85', '+2.78'],
  ['optimal_diagonal_control', 'feedback', '5.62', '5.62', '−0.01', '+0.08', '+0.41'],
  ['optimal_diagonal_control_fast', 'feedback', '5.64', '5.66', '+0.02', '+0.26', '+0.06'],
  ['match_trace_pseudoinverse', 'feedback', '8.22', '7.21', '−1.01', '+0.09', '+0.05'],
  ['buzz_control', 'open', '7.90', '5.91', '−1.99', '+9.38', '−0.30'],
  ['pseudoinverse_control', 'open', '10.47', '7.01', '−3.47', '+13.55', '−1.28'],
];
const th = ['Law', 'Loop', 'rms cap on', 'rms cap off', 'Δ', 'level on', 'level off'];
const t11 = [th.map((h, i) => ({ text: h,
  options: { bold: true, color: INK, fill: { color: PANEL }, align: i > 1 ? 'right' : 'left' } }))];
capRows.forEach(r => t11.push([
  { text: r[0], options: { fontFace: MONO, fontSize: 10.5 } },
  { text: r[1], options: { color: r[1] === 'open' ? ORANGE : BLUE, bold: true } },
  { text: r[2], options: { align: 'right' } },
  { text: r[3], options: { align: 'right' } },
  { text: r[4], options: { align: 'right', bold: Math.abs(parseFloat(r[4])) > 1,
    color: Math.abs(parseFloat(r[4])) > 1 ? ORANGE : INK } },
  { text: r[5], options: { align: 'right' } },
  { text: r[6], options: { align: 'right' } },
]));
s.addTable(t11, { x: M, y: 1.5, w: W - 2 * M, fontFace: BODY, fontSize: 11.5, color: INK,
  border: { type: 'solid', color: 'DDE1E3', pt: 0.5 },
  colW: [3.6, 1.1, 1.5, 1.5, 1.1, 1.45, 1.85], rowH: 0.34 });
const mech = [
  ['1', 'Both open-loop solves are nearly rank one',
   'Eigenvalue participation ratio 1.0–1.1 of 6 — the six shakers act as one source.'],
  ['2', 'The responses cancel by precise inter-drive phasing',
   'That phasing lives entirely in the off-diagonal terms of the drive CPSD.'],
  ['3', '_cap_drive_coherence shrinks exactly those terms',
   'Diagonal preserved, so drive power barely moves — but the cancellation is destroyed and the level jumps ~20 dB.'],
];
let yy = 4.16;
mech.forEach(([n, h, sub]) => {
  s.addShape(pres.ShapeType.ellipse, { x: M, y: yy + 0.03, w: 0.4, h: 0.4, fill: { color: ORANGE } });
  s.addText(n, { x: M, y: yy + 0.03, w: 0.4, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: 'FFFFFF',
    align: 'center', valign: 'middle', margin: 0 });
  s.addText(h, { x: M + 0.58, y: yy, w: W - 2 * M - 0.58, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addText(sub, { x: M + 0.58, y: yy + 0.3, w: W - 2 * M - 0.58, h: 0.32, isTextBox: true,
    fontFace: BODY, fontSize: 12, color: INK2, margin: 0 });
  yy += 0.72;
});
note(s, M, 6.16, W - 2 * M, 0.78, 'For pseudoinverse_control and buzz_control, cap off (1.0) is the CORRECT setting',
  'Nothing in the parameter name or the docstrings warns of it. The best achievable drive breaches 0.95 on 49 % of its pairs.', ORANGE);
s.addNotes(
  'Applied to a feedback law the cap is a constraint the loop re-converges '
  + 'around. Applied to an open-loop law it is pure damage, because there is '
  + 'nothing to re-converge with.\n\n'
  + 'The residual is shape error, not level: the best possible single rescale of '
  + 'run 09\'s entire drive recovers only 0.30 dB. No gain correction rescues it.\n\n'
  + 'match_trace is the one feedback law that loses a full dB to the cap — but '
  + 'removing it costs 50x the drive power, so cap on is still right there.');

/* ------------------------------------------------- 12. choosing / defects */
s = pres.addSlide();
s.background = { color: DARK };
s.addText('Choosing a law', { x: M, y: 0.6, w: W - 2 * M, h: 0.66, isTextBox: true,
  fontFace: HEAD, fontSize: 32, bold: true, color: 'FFFFFF', margin: 0 });
const pick = [
  ['Lowest error against the specification', 'match_diagonal_congruence, cap on or off', '5.23 dB', GREEN],
  ['Fewest lines outside ±6 dB', 'optimal_diagonal_control_fast', '10.7 %', GREEN],
  ['Least drive power', 'match_diagonal_congruence, cap on', '0.047 V²', GREEN],
  ['Simplest thing that works', 'buzz_control, cap OFF', '5.91 dB, no feedback', ORANGE],
  ['Do not use as-is', 'pseudoinverse_control with the cap on', '10.47 dB, +13.6 dB high', 'E05B5B'],
];
yy = 1.64;
pick.forEach(([k, v, n, c]) => {
  s.addText(k, { x: M, y: yy, w: 4.6, h: 0.42, isTextBox: true,
    fontFace: BODY, fontSize: 14, color: 'B9C5CC', margin: 0, valign: 'middle' });
  s.addText(v, { x: M + 4.8, y: yy, w: 4.7, h: 0.42, isTextBox: true,
    fontFace: MONO, fontSize: 12.5, color: 'FFFFFF', margin: 0, valign: 'middle' });
  s.addText(n, { x: M + 9.7, y: yy, w: W - M - (M + 9.7), h: 0.42, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: c, margin: 0, valign: 'middle', align: 'right' });
  yy += 0.62;
});
s.addText('Open defects', { x: M, y: 4.94, w: W - 2 * M, h: 0.4, isTextBox: true,
  fontFace: HEAD, fontSize: 19, bold: true, color: ORANGE, margin: 0 });
s.addText([
  { text: 'No startup level cap on either optimal-diagonal law, or on pseudoinverse_control.', options: { bullet: true, breakLine: true } },
  { text: 'The fast law\'s Burer-Monteiro path never engages with FRF update on — its central claim is untested.', options: { bullet: true, breakLine: true } },
  { text: 'The optimal-diagonal coherence cap reaches only 20 of 901 bins per call.', options: { bullet: true, breakLine: true } },
  { text: 'buzz_control_generator and buzz_control_class have neither the cap nor the startup guard; the class ignores last_response_cpsd entirely.', options: { bullet: true } },
], { x: M, y: 5.38, w: W - 2 * M, h: 1.6, isTextBox: true,
     fontFace: BODY, fontSize: 12.5, color: 'B9C5CC', margin: 0, paraSpaceAfter: 5 });
s.addNotes(
  'The honest summary: on this plant, against a flat specification, the four '
  + 'feedback laws and the better of the two open-loop laws all land between '
  + '5.2 and 5.9 dB. That cluster is set by reachability and identification '
  + 'quality, not by loop design.\n\n'
  + 'What actually separates them is how sensibly each picks a target inside '
  + 'the reachable set — which is why buzz, with no feedback at all, beats '
  + 'match_trace, which has an integrator.');

pres.writeFile({ fileName: 'control_law_reference_deck.pptx' })
  .then(f => console.log('wrote', f));
