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
idl = [r['ideal_rms'] for r in rows]
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

<h2>1. The system under test</h2>

<p>Six drives, twelve responses, eight control channels, controlled over
100–1000&nbsp;Hz against a flat 1&times;10<sup>-3</sup> specification with zero
cross-terms. The plant is the linear SDynPy frame model
<span class="mono">sdynpy_frame6x12_system.npz</span> (hardware type 6, 36 DOF),
run as virtual hardware at 4096&nbsp;Hz.</p>

{img('system_characterisation.png',
     'Left: modal damping for the 33 modes inside the control band — lightly '
     'damped, median 0.83 %, minimum 0.70 %. Centre: the reachability floor per '
     'run, at the rcond the laws actually use and with full-rank drive; the gap '
     'between the pair is the price of the rcond = 1e-3 truncation. Right: '
     'median cond(H) per run — the same deterministic plant, identified with '
     'identical settings, scattering by a factor of ten.')}

<p><strong>The reachability limit is structural, not a control failure.</strong>
With M&nbsp;=&nbsp;8 control channels and N&nbsp;=&nbsp;6 drives, the set of
achievable response diagonals
<span class="mono">D = {{ diag(H X H<sup>H</sup>) : X ⪰ 0 }}</span> is a proper
subset of the nonnegative orthant. Part of the flat specification is
unreachable by any law. Measured on this plant, per run, from that run's own
FRF: <strong>{min(fl):.2f}–{max(fl):.2f}&nbsp;dB rms</strong> at
<span class="mono">rcond = 1e-3</span>.</p>

<div class="note"><strong>New: most of the floor is not fundamental</strong>
Recomputing the same floor without the rcond restriction — that is, allowing
full-rank drive — gives <strong>{min(idl):.2f}–{max(idl):.2f}&nbsp;dB</strong>.
So roughly 1.6&nbsp;dB of what has been called "unreachable" is not a property
of the plant at all; it is the <span class="mono">rcond = 1e-3</span>
truncation every one of these laws inverts with. Nobody has tried moving it.
That is probably the cheapest available improvement in this whole campaign.</div>

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

<h2>4. The coherence cap inverts by family</h2>

{img('cap_effect.png',
     'Each law at cap 0.95 (hollow) and cap off (solid). Colour is loop type. '
     'The open-loop laws gain 2.0 and 3.5 dB from removing the cap; three of '
     'the four feedback laws move by 0.08 dB or less.')}

<div class="tw"><table>
<thead><tr><th>Law</th><th>Loop</th><th>rms cap on</th><th>rms cap off</th>
<th>Δ</th><th>level on</th><th>level off</th><th>drive on</th><th>drive off</th>
</tr></thead>
<tbody>
{cap_rows()}
</tbody></table></div>

<p>The mechanism is visible in the drive solution. Both open-loop laws produce
a strongly rank-concentrated drive — eigenvalue participation ratio 1.0–1.1 out
of 6 — so the six shakers act as essentially one source and the responses
cancel by precise inter-drive phasing.
<span class="mono">_cap_drive_coherence</span> shrinks the off-diagonal terms
while preserving the diagonal, which breaks that cancellation: the drive power
is barely changed, but the responses now add incoherently and the level jumps.
Run 09 sat <strong>+13.55&nbsp;dB</strong> above specification and run 11
<strong>+9.38&nbsp;dB</strong>; uncapped, the same laws land at −1.28 and
−0.30&nbsp;dB.</p>

<p>A feedback law survives the same treatment because its loop re-converges
around the cap as a constraint. An open-loop law has nothing to re-converge
with, so the cap is pure damage. The residual is <em>shape</em> error, not
level error: the best possible single rescale of run 09's whole drive improves
it from 10.47 to only 10.17&nbsp;dB.</p>

<div class="note"><strong>Operational consequence</strong>
For <span class="mono">pseudoinverse_control</span> and
<span class="mono">buzz_control</span>, cap off (1.0) is the correct setting,
not the reckless one. This is the reverse of the feedback-law intuition, and
nothing in the parameter name or the docstring warns of it.</div>

<h2>5. Feedback buys less than expected</h2>

{img('rms_vs_pout.png',
     'The two acceptance criteria against each other, marker area proportional '
     'to log drive power. Hollow markers are cap-on runs. At their correct cap '
     'setting the open-loop laws sit inside the feedback cluster.')}

<p><span class="mono">buzz_control</span> at 5.91&nbsp;dB / 13.6&nbsp;% finishes
0.68&nbsp;dB behind the best run in the matrix and ahead of both
<span class="mono">match_trace_pseudoinverse</span> runs — with no error
feedback whatsoever. Tracing the code confirms it:
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
by reachability and identification quality, not by loop design. What separates
the laws is how sensibly they choose a target inside the reachable set.
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

<p>Six of the eight control channels sit within a couple of dB under every law.
13X+ and 14X+ are the two directions six shakers cannot independently command,
and every law undershoots them: −2.2/−2.8&nbsp;dB for congruence, −5.3/−6.0 for
buzz, −6.7/−7.3 for match_trace, −8.5/−9.2 for pseudoinverse. The laws differ
mainly in how much drive they are willing to waste trying.</p>

<h2>8. Drive power</h2>

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

<h2>9. Caveats</h2>

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

<h2>10. Next</h2>

<ol>
<li><strong>Move rcond.</strong> The floor at full-rank drive is 1.6&nbsp;dB
below the floor at <span class="mono">rcond = 1e-3</span>. That is a larger prize
than anything separating the six laws, and it costs one parameter sweep.</li>
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
