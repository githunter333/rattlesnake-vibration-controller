"""Assemble the self-contained HTML report for the 6-12-8 law comparison.

Revision 2026-09-14: twelve runs (was eight).  Adds pseudoinverse_control and
buzz_control, the two original upstream open-loop laws, each with the drive
coherence cap on and off.
"""
import base64, json, os

FIGS = 'figs'
rows = json.load(open('report_data.json'))
by = {r['run']: r for r in rows}
STYLE = open('_style_block.html').read()
CT = json.load(open('crossterm_data.json'))
SV = json.load(open('singular_value_data.json'))
CTF = CT['floor']
CTR = sorted(CT['runs'], key=lambda r: r['resp_med'])

NICE = {'match_trace': 'match_trace_pseudoinverse',
        'optimal_diagonal': 'optimal_diagonal_control',
        'optimal_diagonal_fast': 'optimal_diagonal_control_fast',
        'congruence': 'match_diagonal_congruence',
        'pseudoinverse': 'pseudoinverse_control', 'buzz': 'buzz_control'}


def img(name, caption):
    b = base64.b64encode(open(os.path.join(FIGS, name), 'rb').read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


best_rms = min(r['rms'] for r in rows)
best_out = min(r['pout'] for r in rows)
best_ach = min(r['ach_rms'] for r in rows)


def result_rows():
    out = []
    for r in sorted(rows, key=lambda x: x['run']):
        out.append(f"""<tr>
<td class="mono">{r['run']}</td>
<td class="mono" style="font-size:11.5px">{NICE[r['law']]}</td>
<td>{'open' if r['loop'] == 'open' else 'feedback'}</td>
<td>{r['cap']}</td>
<td class="num{' win' if r['rms'] == best_rms else ''}">{r['rms']:.2f}</td>
<td class="num{' win' if r['pout'] == best_out else ''}">{r['pout']:.1f}</td>
<td class="num{' win' if r['ach_rms'] == best_ach else ''}">{r['ach_rms']:.2f}</td>
<td class="num">{r['reach_rms']:.2f}</td>
<td class="num">{r['drive_trace']:.4g}</td>
<td class="num">{r['level']:+.2f}</td>
<td class="num">{r['cond']:,.0f}</td>
<td class="num">{r['min_mult_coh']:.3f}</td></tr>""")
    return '\n'.join(out)


def sv_rows():
    out = []
    for a, m in zip(SV['analytic'], SV['measured']):
        null = a['i'] == 6
        out.append(f"""<tr>
<td class="mono">σ{a['i']}</td>
<td class="num">{a['lo']:.2e}</td><td class="num">{a['med']:.2e}</td>
<td class="num">{a['hi']:.2e}</td>
<td class="num{'' if null else ' win'}">{a['db']:.1f}</td>
<td class="num">{m['med']:.2e}</td><td class="num">{m['db']:.1f}</td></tr>""")
    return '\n'.join(out)


def crossterm_rows():
    out = []
    for r in CTR:
        out.append(f"""<tr>
<td class="mono">{r['run']}</td>
<td>{NICE[{'match trace':'match_trace','opt diagonal':'optimal_diagonal',
            'opt diag fast':'optimal_diagonal_fast','congruence':'congruence',
            'pseudoinverse':'pseudoinverse','buzz':'buzz'}[r['law']]]}</td>
<td>{r['loop']}</td><td>{r['cap']}</td>
<td class="num">{r['resp_med']:.3f}</td><td class="num">{r['resp_p90']:.3f}</td>
<td class="num">{r['part']:.2f}</td>
<td class="num">{r['drive_med']:.3f}</td><td class="num">{r['drive_over_cap']:.1f}</td>
</tr>""")
    return '\n'.join(out)


def cap_rows():
    fams = ['congruence', 'optimal_diagonal', 'optimal_diagonal_fast',
            'match_trace', 'buzz', 'pseudoinverse']
    out = []
    for f in fams:
        on = next(r for r in rows if r['law'] == f and r['cap'] != 'off')
        off = next(r for r in rows if r['law'] == f and r['cap'] == 'off')
        d = off['rms'] - on['rms']
        out.append(f"""<tr>
<td class="mono" style="font-size:11.5px">{NICE[f]}</td>
<td>{'open' if on['loop'] == 'open' else 'feedback'}</td>
<td class="num">{on['rms']:.2f}</td><td class="num">{off['rms']:.2f}</td>
<td class="num{' win' if d < -1 else ''}">{d:+.2f}</td>
<td class="num">{on['level']:+.2f}</td><td class="num">{off['level']:+.2f}</td>
<td class="num">{on['drive_trace']:.3g}</td>
<td class="num">{off['drive_trace']:.3g}</td></tr>""")
    return '\n'.join(out)


fl = [r['reach_rms'] for r in rows]
idl = [r['rank_rms'] for r in rows]
sv6 = [r['sv6_ratio'] for r in rows]
cond = [r['cond'] for r in rows]
dr = [r['drive_trace'] for r in rows]

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>6-12-8 Control Law Comparison</title>
{STYLE}
</head>
<body>
<div class="wrap">

<h1>Control-law comparison on the 6-12-8 system</h1>
<p class="sub">Twelve runs · linear plant · 14 September 2026 ·
<span class="mono">rattlesnake-vibration-controller</span>, branch
<span class="mono">control-law-development-2026-09</span></p>

<p class="lede">Six control laws, each run with the drive-coherence cap on and
off, on one plant with every other setting held fixed and machine-verified.
Two findings changed the picture since the eight-run revision:
<strong>the coherence cap means opposite things to feedback and open-loop
laws</strong>, and <strong>a law with no error feedback at all finishes within
0.7&nbsp;dB of the best feedback law</strong>.</p>

<div class="note"><strong>What changed since 12 September</strong>
Runs 09–12 are new: <span class="mono">pseudoinverse_control</span> and
<span class="mono">buzz_control</span>, the two original upstream laws, each
capped and uncapped. Every number in this report was recomputed from the run
files with one method, so the eight earlier rows may differ in the last digit
from the previous revision. The achievable-floor column is unchanged in
meaning and reproduces the earlier values (run01 3.88 here, 3.87 there — the
difference is floor decimation 5 versus 4).</div>

<div class="note"><strong>Correction to the first draft of this report</strong>
An earlier version of this document claimed that roughly 1.6&nbsp;dB of the
reachability floor was an artefact of the
<span class="mono">rcond = 1e-3</span> truncation, and listed relaxing rcond as
the cheapest available improvement. <strong>That was wrong.</strong> The
unrestricted floor it rested on (1.96–2.37&nbsp;dB) was the optimizer
exploiting a sixth singular direction that exists only as identification noise.
Section 1 has the evidence. There is no headroom there, and the
recommendation has been withdrawn.</div>

<h2>1. The system under test</h2>

<p>Six drives, twelve responses, eight control channels, controlled over
100–1000&nbsp;Hz against a flat 1&times;10<sup>-3</sup> specification with zero
cross-terms. The plant is the linear SDynPy frame model
<span class="mono">sdynpy_frame6x12_system.npz</span> (hardware type 6, 36 DOF),
run as virtual hardware at 4096&nbsp;Hz.</p>

{img('system_characterisation.png',
     'Left: modal damping for the 33 modes inside the control band — lightly '
     'damped, median 0.83 %, minimum 0.70 %. Centre: the reachability floor per '
     'run, at the rcond the laws actually use and unrestricted over the plant’s '
     'real rank-5 subspace; the two agree to 0.21 dB, which is why rcond = 1e-3 '
     'costs nothing. Right: median cond(H) per run — the same deterministic '
     'plant, identified with identical settings, scattering by a factor of ten.')}

<p><strong>The reachability limit is structural, not a control failure.</strong>
With M&nbsp;=&nbsp;8 control channels and N&nbsp;=&nbsp;6 drives, the set of
achievable response diagonals
<span class="mono">D = {{ diag(H X H<sup>H</sup>) : X ⪰ 0 }}</span> is a proper
subset of the nonnegative orthant. Part of the flat specification is
unreachable by any law. Measured on this plant, per run, from that run's own
FRF: <strong>{min(fl):.2f}–{max(fl):.2f}&nbsp;dB rms</strong> at
<span class="mono">rcond = 1e-3</span>.</p>

<h3>The plant is rank 5, not 6</h3>

{img('singular_values.png',
     'Singular values of the 8×6 FRF over the control band. Left: the measured '
     'in-control FRF, the matrix every law actually inverts. Right: median and '
     '10th-90th percentile of each singular value relative to σ1, analytic '
     'against measured.')}

<div class="tw"><table>
<thead><tr><th></th><th colspan="4" style="text-align:center">analytic (exact, from M, K, C)</th>
<th colspan="2" style="text-align:center">measured</th></tr>
<tr><th></th><th>min</th><th>median</th><th>max</th><th>dB re σ1</th>
<th>median</th><th>dB re σ1</th></tr></thead>
<tbody>
{sv_rows()}
</tbody></table></div>

<p>Units are (m/s²)/N over {SV['n_lines']} lines. σ1 through σ5 are genuine and
well separated — about 10&nbsp;dB between the first two, then 8–11&nbsp;dB steps,
spanning {abs(SV['analytic'][4]['db']):.0f}&nbsp;dB in all — and the analytic and
measured values agree to a few tenths of a dB throughout. σ6 does not.</p>

<div class="note"><strong>Quote σ1/σ5, not σ1/σ6</strong>
The useful conditioning of this plant is
<strong>σ1/σ5 = {SV['cond_15_analytic']:.0f}</strong> analytic,
{SV['cond_15_measured']:.0f} measured — benign for inversion. The
<span class="mono">cond(H)</span> column in section 3 is σ1/σ6, median
{SV['cond_16_measured']:,.0f}, and it is measuring the identification noise
floor rather than the plant. That is why it scatters by a factor of ten across
twelve identifications of one deterministic system: the noise floor moves, the
plant does not.</div>

<p>Computing H analytically from the model's mass, stiffness and damping
matrices — validated against the measured FRF, whose magnitude it matches to a
median of 0.21&nbsp;dB and whose first five singular values it reproduces to a
few percent — the <strong>sixth singular value is ~1&times;10<sup>-17</sup></strong>.
The six shakers span only a five-dimensional response subspace at these eight
control locations. The sixth drive direction buys nothing.</p>

<p>In the measured FRF that direction comes back at
{min(sv6):.1e}–{max(sv6):.1e} of the first singular value, and the numeric rank
over the control band is 5 in <strong>all twelve runs</strong>. That residue is
identification noise in a direction the hardware cannot reach.</p>

<div class="note"><strong>Consequence: rcond = 1e-3 is correct, and costs nothing</strong>
Recomputing each floor with no rcond restriction but over the plant's real
rank-5 subspace gives {min(idl):.2f}–{max(idl):.2f}&nbsp;dB, within
<strong>0.21&nbsp;dB</strong> of the operational floor at every run. The same
computation on the raw measured FRF returns ~2.3&nbsp;dB — but that number is
the optimizer spending drive on the noise direction, and it is not achievable.
Lowering rcond would invert noise, not recover headroom.</div>

<h2>2. What was run</h2>

<p>Every run used the same plant, sample rate, averaging, and system-ID
configuration. Twenty-three settings are compared across runs by
<span class="mono">score_runs_sdynpy.py</span>'s
<span class="mono">MUST_MATCH</span> list and a mismatch raises a
NOT-COMPARABLE warning rather than being scored silently — a response to an
earlier campaign where sixteen runs turned out to span three different plants.</p>

<pre>samples per frame   4096          control averaging    Exponential, 0.04
sample rate         4096 Hz       frames in CPSD       20
time per read/write 0.25 s        update sys ID        Y (on)
sys ID averages     20            sys ID level         1.0 V rms
control band        100-1000 Hz   rcond                1e-3</pre>

<h2>3. Results</h2>

<div class="tw"><table>
<thead><tr>
<th>Run</th><th>Law</th><th>Loop</th><th>Cap</th>
<th>rms dB</th><th>% out</th><th>vs floor</th><th>floor</th>
<th>drive V²</th><th>level dB</th><th>cond(H)</th><th>min MC</th>
</tr></thead>
<tbody>
{result_rows()}
</tbody></table></div>

<p><span class="mono">rms dB</span> and <span class="mono">% out</span> score
the run against the flat specification. <span class="mono">vs floor</span>
scores it against what this plant could actually deliver, and is the figure
that isolates the law's own contribution. <span class="mono">floor</span> is
how far that best-achievable response itself sits from the specification.
<span class="mono">min MC</span> is the minimum multiple coherence of the
in-control FRF re-identification — a measure of how trustworthy that run's own
plant model was.</p>

<h2>4. What the coherence cap costs, and why</h2>

{img('cap_effect.png',
     'Each law at cap 0.95 (hollow) and cap off (solid). The spread is not '
     'explained by loop type — it is explained by where the cap is applied and '
     'how much authority the law has to recover afterward.')}

<div class="tw"><table>
<thead><tr><th>Law</th><th>Loop</th><th>rms cap on</th><th>rms cap off</th>
<th>Δ</th><th>level on</th><th>level off</th><th>drive on</th><th>drive off</th>
</tr></thead>
<tbody>
{cap_rows()}
</tbody></table></div>

<div class="note"><strong>Correction, 2026-09-17</strong>
An earlier version of this section attributed the split to loop type — feedback
laws tolerate the cap, open-loop laws do not. <strong>That was wrong.</strong>
<span class="mono">optimal_diagonal_control</span> and its fast subclass never
read <span class="mono">last_response_cpsd</span>: they refine each bin against
their own <em>predicted</em> response and are open-loop, yet the cap costs them
nothing. Only <span class="mono">match_trace_pseudoinverse</span> and
<span class="mono">match_diagonal_congruence</span> close a loop at all. The
explanation below replaces it.</div>

<p><strong>Where the cap is applied is what matters.</strong> The
optimal-diagonal laws solve <em>subject to</em> the cap — it is a cvxpy
constraint inside the SDP
(<span class="mono">optimal_diagonal_control.py:248</span>,
<span class="mono">cp.abs(X[i,j]) &lt;= max_drive_coherence·√(X_ii·X_jj)</span>),
so whatever they return is already optimal given the constraint and there is
nothing to lose. Every other law calls
<span class="mono">_cap_drive_coherence</span> on an
<em>already-computed</em> solve, which is a different and much more damaging
operation.</p>

<p><strong>How much the post-process costs then depends on what the law can do
about it.</strong> The pseudoinverse-family solves are strongly
rank-concentrated — eigenvalue participation 1.0–1.1 out of 6 — so the six
shakers act as essentially one source and the responses cancel by precise
inter-drive phasing. That phasing lives entirely in the off-diagonal terms.
<span class="mono">_cap_drive_coherence</span> shrinks them while preserving
the diagonal, so drive power barely changes but the cancellation is destroyed
and the responses add incoherently. What happens next is the whole story:</p>

<div class="tw"><table>
<thead><tr><th>Law</th><th>Cap applied</th><th>Correction authority after it</th><th>Δ from cap off</th></tr></thead>
<tbody>
<tr><td class="mono">optimal_diagonal_control</td><td>SDP constraint</td><td>— (already optimal under it)</td><td class="num">−0.01</td></tr>
<tr><td class="mono">optimal_diagonal_control_fast</td><td>SDP constraint</td><td>— (already optimal under it)</td><td class="num">+0.02</td></tr>
<tr><td class="mono">match_diagonal_congruence</td><td>post-process</td><td>per-drive log error, re-converges</td><td class="num">−0.08</td></tr>
<tr><td class="mono">match_trace_pseudoinverse</td><td>post-process</td><td>one scalar — level only</td><td class="num">−1.01</td></tr>
<tr><td class="mono">buzz_control</td><td>post-process</td><td>none</td><td class="num">−1.99</td></tr>
<tr><td class="mono">pseudoinverse_control</td><td>post-process</td><td>none</td><td class="num">−3.47</td></tr>
</tbody></table></div>

<p>Congruence corrects per drive and re-converges around the damage, so it
barely notices. match_trace can only rescale the whole matrix, which cannot
undo a shape change, so it loses a full dB. The two laws with no error
feedback at all cannot recover anything, and pay 2.0 and 3.5&nbsp;dB. The
residual is <em>shape</em> error, not level: the best possible single rescale
of run 09's whole drive improves it from 10.47 to only 10.17&nbsp;dB — which
is precisely why match_trace's scalar is not enough.</p>

<p>The level offsets make it starker than rms does. Run 09 sat
<strong>+13.55&nbsp;dB</strong> above specification with the cap on and
−1.28&nbsp;dB with it off; run 11 went from +9.38 to −0.30&nbsp;dB.</p>

<div class="note"><strong>Operational consequence</strong>
For <span class="mono">pseudoinverse_control</span> and
<span class="mono">buzz_control</span>, cap off (1.0) is the correct setting,
not the reckless one — and for the optimal-diagonal laws the cap is close to
free, so leave it on. Nothing in the parameter name or the docstrings conveys
either fact.</div>

<h2>5. Only two of the six laws close a loop at all</h2>

{img('rms_vs_pout.png',
     'The two acceptance criteria against each other, marker area proportional '
     'to log drive power. Hollow markers are cap-on runs. At their correct cap '
     'setting the open-loop laws sit inside the feedback cluster.')}

<p>Reading the six laws for what they actually do with
<span class="mono">last_response_cpsd</span> gives a different census than this
report carried until 2026-09-17. <strong>Only
<span class="mono">match_trace_pseudoinverse</span> and
<span class="mono">match_diagonal_congruence</span> read it.</strong>
<span class="mono">optimal_diagonal_control</span> and its fast subclass accept
it and never reference it — they refine each bin against
<span class="mono">diag(H X H<sup>H</sup>)</span>, their own prediction, so
they adapt to a changing plant model rather than to their own error. Four of
six laws are open-loop.</p>

<p>That makes the original observation stronger rather than weaker: the top of
the table is now almost entirely open-loop.
<span class="mono">buzz_control</span> at 5.91&nbsp;dB / 13.6&nbsp;% finishes
0.68&nbsp;dB behind the best run in the matrix and ahead of both
<span class="mono">match_trace_pseudoinverse</span> runs, and the
optimal-diagonal family at 5.62–5.66&nbsp;dB is open-loop too. Tracing the code
confirms it:
<span class="mono">last_response_cpsd</span> appears in the signature and the
docstring and never in the body; <span class="mono">last_output_cpsd</span> is
read only as a first-call sentinel for the startup clamp. The law recomputes
the same open-loop solve every cycle.</p>

<p>Its one adaptive channel is the plant model. With
<span class="mono">update_tf_during_control = Y</span> the FRF is re-identified
from live control data, but <span class="mono">sysid_response_cpsd</span> — the
source of the coherence and phase it matches — is assigned only during system
ID and is frozen for the whole run. So the law adapts to a changing plant, not
to its own error, which is why runs 11 and 12 sat at a steady offset instead of
converging toward 0&nbsp;dB.</p>

<p>The reading this supports: on this plant the 5.2–5.9&nbsp;dB cluster is set
by reachability and identification quality, not by loop design — and with four
of the six laws open-loop, there was never much loop design in it to begin
with. What separates the laws is how sensibly they choose a target inside the
reachable set.
<span class="mono">buzz_control</span>'s substitution of measured coherence and
phase for the specification's unreachable diagonal is worth 1.1&nbsp;dB and a
third of the drive against <span class="mono">pseudoinverse_control</span>,
which inverts the raw diagonal target and pays for it.</p>

<h2>6. Separating the law's error from the plant's limit</h2>

{img('error_vs_floor.png',
     'Two independent scorings of each run. The bar is error against the '
     "achievable floor - the law's own contribution. The diamond is total "
     'error against the flat specification. These are separate rms figures and '
     'do not subtract.')}

<p>Ranked against the floor, <span class="mono">match_diagonal_congruence</span>
leads at 3.48–3.51&nbsp;dB, the optimal-diagonal family follows at a flat
3.99&nbsp;dB, <span class="mono">buzz_control</span> at 4.20, then
<span class="mono">pseudoinverse_control</span> at 5.48 and
<span class="mono">match_trace_pseudoinverse</span> at 5.78. The ordering is not
the same as the flat-spec ordering — congruence sits +2.8&nbsp;dB high in level,
which costs it against the specification but not against the floor.</p>

<h2>7. The same two channels dominate every law</h2>

{img('per_channel_mean.png',
     'Mean per-channel error, each law at its own best cap setting, with the '
     'reachability floor behind. Channels 13X+ and 14X+ are the '
     'reachability-limited pair.')}

<p>Five of the eight control channels — 8X+ through 12X+ — stay within
2.1&nbsp;dB under every law. 7X+ is marginal: inside 2&nbsp;dB for four of the
five laws but −3.6&nbsp;dB under
<span class="mono">pseudoinverse_control</span>.
13X+ and 14X+ are the two directions six shakers cannot independently command,
and every law undershoots them: −2.2/−2.8&nbsp;dB for congruence, −5.3/−6.0 for
buzz, −6.7/−7.3 for match_trace, −8.5/−9.2 for pseudoinverse. The laws differ
mainly in how much drive they are willing to waste trying.</p>

<h2>8. Cross-terms: the specification asked for zero</h2>

<p>The flat specification is diagonal — every off-diagonal term is zero. None
of these six laws controls the off-diagonals; every one of them matches the
<em>diagonal</em> and lets the cross-structure fall out. It does not fall out
small.</p>

{img('cross_terms.png',
     'Cross-channel coherence magnitude across all twelve runs. Box is the '
     'quartile range, whisker the 10th-90th percentile, marker the median. '
     'Left: response, 28 channel pairs over 901 in-band lines. Right: drive, '
     '15 pairs.')}

<div class="tw"><table>
<thead><tr><th>Run</th><th>Law</th><th>Loop</th><th>Cap</th>
<th>resp median |γ|</th><th>resp p90</th><th>eig participation</th>
<th>drive median |γ|</th><th>drive % &gt; 0.95</th></tr></thead>
<tbody>
{crossterm_rows()}
</tbody></table></div>

<p>Median response coherence runs from {min(r['resp_med'] for r in CTR):.3f} to
{max(r['resp_med'] for r in CTR):.3f}, with the 90th percentile above 0.96 for
nine of the twelve runs. In absolute terms the off-diagonal magnitudes sit
within a few dB of the 1&times;10<sup>-3</sup> diagonal — the same order as the
specification itself, not a small perturbation on it.</p>

<h3>Zero cross-terms is unreachable, not merely difficult</h3>

<p>The plant is rank 5 (section 1), so the response CPSD
<span class="mono">H X H<sup>H</sup></span> has rank at most 5 in an
eight-dimensional space. A genuinely diagonal 8&times;8 CPSD has rank 8. The
specification therefore asks for something outside the achievable set for a
second, independent reason beyond the magnitude shortfall on 13X+ and 14X+.</p>

<p>The eigenvalue participation ratios put a number on how far outside. Nine of
the twelve runs sit between 1.15 and 1.5 <em>effective response directions out
of eight</em> — all eight control channels riding essentially one motion
pattern.</p>

<div class="note"><strong>The optimum is the most correlated response of all</strong>
Solving for the drive that best matches the flat diagonal magnitudes and then
measuring <em>its</em> cross-terms gives a median coherence of
<strong>{CTF['resp_med']:.3f}</strong> and a participation ratio of
{CTF['part']:.2f}. The best achievable response is more correlated than almost
every law delivered. Chasing decorrelation moves away from matching the
specification, not toward it.</div>

<p>Run 10 is the control experiment in the other direction. It is the only run
that produced substantially independent responses — median coherence
{CTR[0]['resp_med']:.3f}, participation {CTR[0]['part']:.2f} — and it spent
5.38&nbsp;V² doing it, roughly seventy times the optimal-diagonal family, to
finish at 7.01&nbsp;dB against their 5.62. Decorrelating the response is
expensive and buys nothing here, because nothing in the acceptance criterion
looks at it.</p>

<h3>What the cap does to the drive</h3>

<p>The cap acts on the drive, not the response, and it bites only the upper
tail: median drive coherence moves 0.440 to 0.525 for congruence and 0.576 to
0.652 for optimal diagonal, while the fraction of pairs above 0.95 goes from
roughly 0–3&nbsp;% to 5–10&nbsp;%. For context, the optimal drive found by the
floor solver sits above 0.95 on <strong>{CTF['drive_over_cap']:.0f}&nbsp;%</strong>
of its pairs, median {CTF['drive_med']:.3f} — the cap constrains the solution
away from the optimum by construction, which is section 4's result seen from
the drive side.</p>

<div class="note"><strong>Reading the cap-on runs</strong>
Capped runs still show a few percent of measured drive pairs above 0.95. The
cap applies to the <em>commanded</em> drive CPSD; what is saved and scored here
is the <em>measured</em> one. A small excess is expected and is not evidence the
cap failed.</div>

<p><strong>Consequence for real tests.</strong> If response cross-terms matter
for the article under test — and on a real structure they often do — none of
these laws gives control over them, and no parameter setting changes that. It
needs a law carrying off-diagonal terms in its objective and a specification
that states them. <span class="mono">buzz_control</span> is the only law here
that references measured coherence at all, and it copies the plant's rather
than commanding anything.</p>

<h2>9. Drive power</h2>

{img('drive_power.png',
     'Mean drive trace per run, log scale, coloured by loop type. The spread '
     'across twelve runs is a factor of 128.')}

<p>Drive power spans {min(dr):.4g} to {max(dr):.4g}&nbsp;V², a factor of
{max(dr)/min(dr):.0f}. The open-loop laws are expensive:
<span class="mono">buzz_control</span> uncapped uses 1.47&nbsp;V², about twenty
times the optimal-diagonal family, for a 0.3&nbsp;dB worse score. If drive
headroom is the binding constraint rather than control accuracy, the ordering
changes completely and <span class="mono">match_diagonal_congruence</span> with
the cap on (0.047&nbsp;V²) wins outright.</p>

<h2>10. Caveats</h2>

<ul>
<li><strong>Runs 05 and 06 did not test what they were meant to.</strong>
<span class="mono">optimal_diagonal_control_fast</span> disables its
Burer-Monteiro fast path whenever the FRF changes between calls, because the
unconstrained factored solve drives pairwise drive coherence to exactly 1.0 and
would make a live H1 estimator's reference CPSD singular. With
<span class="mono">update_tf_during_control = 1</span>, H changes every call, so
every solve fell back to the base class's SDP. Runs 05 and 06 are
<strong>repeats of 03 and 04</strong>, and their close agreement is not evidence
that the factored solve matches the SDP. No timing difference was observed at
the rig, consistent with this.</li>

<li><strong>The coherence cap in the optimal-diagonal laws reaches only
SDP-refined bins</strong>, bounded by
<span class="mono">max_bins_per_update = 20</span> per call. Across 901 lines
most bins carry the deliberately uncapped buzz baseline in both runs, which
likely explains the null result there. Raising the bin budget would distinguish
"the cap does not matter to this law" from "the cap barely applied".</li>

<li><strong>pseudoinverse_control has no startup guard.</strong> It is the only
law in the set without the first-cycle clamp added 2026-09-02. On frame one the
FRF is a single unaveraged estimate and the commanded level can be decades high.
Runs 09 and 10 completed without tripping, but this is luck of the
identification, not safety.</li>

<li><strong>buzz_control_generator and buzz_control_class have neither the cap
nor the startup guard</strong>, and <span class="mono">buzz_control_class.control()</span>
accepts <span class="mono">last_response_cpsd</span> and never references it —
it replays one open-loop solve forever. Both were excluded from this comparison
deliberately.</li>

<li><strong>Run 10 has the second-worst in-control FRF in the matrix</strong>
(minimum multiple coherence 0.525, behind only run 02's 0.275). Both are the
uncapped, near-coherent-drive cases, where the H1 re-identification is fed a
drive with little independent content. Run 10's 7.01&nbsp;dB rests on the least
trustworthy plant model here.</li>

<li><strong>Every floor in this report is a rank-5 floor.</strong> The
computation restricts the drive either by <span class="mono">rcond</span> or by
explicit truncation; both land in the same place because the plant really is
rank 5. Any floor computed on a raw measured FRF without one of those
restrictions will come back optimistically low by about 1.6&nbsp;dB, for the
reason given in section 1. This bit the first draft of this report.</li>

<li><strong>Identification scatter is large.</strong> Median cond(H) came out
{min(cond):,.0f}–{max(cond):,.0f} across twelve identifications of the same
deterministic plant with identical settings — a factor of ten. Differences
between laws smaller than a few tenths of a dB should not be read as real. This
is why the achievable floor is recomputed per run rather than assumed common.</li>

<li><strong>Run 01 may have been saved short of convergence.</strong> Its
optimal single rescale is +3.37&nbsp;dB, which would improve it from 8.22 to
7.50&nbsp;dB rms — the only run in the matrix with a large one-sided level
offset under a law that integrates. It does not change any ordering, but the
match_trace cap-on row is the weakest number here.</li>

<li><strong>Neither optimal-diagonal law has a startup level cap.</strong>
match_trace, buzz and congruence all take
<span class="mono">startup_test_level_cap_db</span> (−9&nbsp;dB default); the
optimal-diagonal laws solve against the full specification and command it
immediately. Worked around by ramping the test level. Still unfixed.</li>

<li><strong>bm_rank was set to 6, not its default of 4.</strong> With eight
control channels the threshold below which spurious local minima generically
appear is √(2·8) ≈ 4, so the default sits exactly on it. Moot given the fast
path never engaged.</li>

<li><strong>The profiles carry no warning or abort levels</strong> — those
columns are all NaN — so the ±6&nbsp;dB tolerance bands are synthesized from the
specification. Putting real tolerances in the profile would make these numbers
contractual rather than nominal.</li>

<li><strong>One test level, one averaging configuration, one plant, no
repeats.</strong> No per-law variance estimate exists. The plant is lightly
damped (median ζ 0.83&nbsp;%), which is the harder case; a better-damped
structure would likely narrow the spread between laws.</li>

<li><strong>An offline predictor was used during the campaign and was
systematically wrong in magnitude.</strong> Reconstructing each law's solve from
a frozen FRF predicted 15.73&nbsp;dB for run 09 (actual 10.47), 12.28 for run 11
(actual 7.90) and 3.87 for run 12 (actual 5.91). It called every ordering
correctly and no magnitude within 2&nbsp;dB. The frozen-FRF assumption is the
reason: the live laws re-invert an updating estimate.</li>

<li><strong>Floor method.</strong> <span class="mono">achievable_diagonal</span>
on each run's own control FRF, every 5th in-band line, log-interpolated,
<span class="mono">restrict_rcond = 1e-3</span>. Decimation was checked against
every-2nd and every-20th line and moves the result by 0.05&nbsp;dB or less.</li>
</ul>

<h2>11. Next</h2>

<ol>
<li><strong>Find out why the plant is rank 5.</strong> If the sixth shaker is
redundant by construction rather than by accident of this model, that matters
for the rig and not just the simulation — it would mean one drive channel is
doing nothing, and that the 8&times;6 problem has always been 8&times;5. The
analytic FRF makes this cheap to investigate.</li>
<li><strong>Higher damping, same geometry.</strong>
<span class="mono">sdynpy_frame6x12_system_higherz.npz</span> has identical mass
and stiffness — mode frequencies match the nominal plant to 0.0000&nbsp;Hz — with
ζ raised from a 0.83&nbsp;% median to 3.34&nbsp;%. Its floor is 4.05&nbsp;dB
against the nominal 4.09, so damping does not change what is reachable; it
halves the conditioning over the real directions (68 to 33). That isolates the
identification-quality hypothesis this report's caveats raise.</li>
<li><strong>buzz_control with the FRF update off.</strong> That makes the law
fully static after cycle one, so the difference against run 12 measures exactly
what plant-model adaptation is worth with nothing else varying. One run.</li>
<li><strong>Raise <span class="mono">max_bins_per_update</span></strong> and
repeat runs 03/04, to settle whether the optimal-diagonal cap null result is
real.</li>
<li><strong>Rerun the fast law with the FRF update off</strong>, so the
Burer-Monteiro path actually engages and runs 05/06 test what they claim to.</li>
<li><strong>Phase two — the achievable target as specification.</strong>
<span class="mono">make_achievable_spec.py</span> is built and verified but
still unused. Pointing every law at a reachable target removes reachability from
the comparison so the residual is tracking behaviour alone.</li>
<li><strong>Add a startup level cap to both optimal-diagonal laws</strong>, and
to <span class="mono">pseudoinverse_control</span>. Open defect.</li>
<li><strong>Decide whether response cross-terms are in scope at all.</strong>
Section 8 shows they are large, uncontrolled, and structurally impossible to
zero on this plant. If they matter for a real article the campaign needs a law
with off-diagonal terms in its objective and a specification that states them;
if they do not, that should be written down so the diagonal-only comparison is
understood as deliberate rather than accidental.</li>
</ol>

<p class="foot">Generated from <span class="mono">report_data.json</span> by
<span class="mono">build_report.py</span>; figures by
<span class="mono">make_report_figures.py</span>; per-run floors by
<span class="mono">compute_report_data.py</span>. All three in
<span class="mono">examples/sixdrive12resp/code/</span>.</p>

</div>
</body>
</html>
"""

open('law_comparison_report.html', 'w').write(HTML)
print('wrote law_comparison_report.html',
      f'{os.path.getsize("law_comparison_report.html")/1024:.0f} KB')
