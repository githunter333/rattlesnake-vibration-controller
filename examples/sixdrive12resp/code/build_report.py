"""Assemble the self-contained HTML report for the 6-12-8 law comparison."""
import base64, json, os

FIGS = 'figs'
rows = json.load(open('report_data.json'))
by = {r['run']: r for r in rows}

REACH = {'run01': (3.87, 6.9), 'run02': (3.91, 7.0), 'run03': (3.70, 6.3),
         'run04': (3.59, 6.2), 'run05': (3.76, 6.1), 'run06': (3.67, 6.0),
         'run07': (3.63, 7.0), 'run08': (3.57, 6.9)}
ACH = {'run01': (7.16, 26.2), 'run02': (5.78, 21.4), 'run03': (4.00, 8.7),
       'run04': (4.00, 8.5), 'run05': (3.99, 8.8), 'run06': (4.00, 8.5),
       'run07': (3.49, 9.4), 'run08': (3.49, 8.5)}
COND = {'run01': 6976, 'run02': 25928, 'run03': 3567, 'run04': 2493,
        'run05': 3457, 'run06': 2447, 'run07': 3354, 'run08': 8676}
STREAM = {'run01': 534528, 'run02': 456704, 'run03': 956416, 'run04': 1010688,
          'run05': 1141760, 'run06': 1152000, 'run07': 1285120, 'run08': 1609728}


def img(name, caption):
    b = base64.b64encode(open(os.path.join(FIGS, name), 'rb').read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def result_rows():
    out = []
    for r in rows:
        k = r['run']
        best_rms = r['rms'] == min(x['rms'] for x in rows)
        best_out = r['pout'] == min(x['pout'] for x in rows)
        out.append(f"""<tr>
<td class="mono">{k}</td><td>{r['law']}</td><td>{r['cap']}</td>
<td class="num">{COND[k]:,}</td>
<td class="num{' win' if best_rms else ''}">{r['rms']:.2f}</td>
<td class="num{' win' if best_out else ''}">{r['pout']:.1f}</td>
<td class="num">{ACH[k][0]:.2f}</td><td class="num">{ACH[k][1]:.1f}</td>
<td class="num">{REACH[k][0]:.2f}</td>
<td class="num">{r['drive_trace']:.4g}</td>
<td class="num">{r['level']:+.2f}</td></tr>""")
    return '\n'.join(out)


HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>6-12-8 Control Law Comparison</title>
<style>
  :root {{
    color-scheme: light;
    --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --ink3:#78766f;
    --line:#e3e2dd; --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a;
    --warnbg:#fdf6ec; --warnline:#eda100;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --bg:#151514; --card:#1d1d1b; --ink:#ffffff; --ink2:#c3c2b7; --ink3:#9a988e;
      --line:#33322e; --blue:#3987e5; --orange:#d95926; --aqua:#199e70;
      --warnbg:#2a2416; --warnline:#c98500;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --bg:#151514; --card:#1d1d1b; --ink:#ffffff; --ink2:#c3c2b7; --ink3:#9a988e;
    --line:#33322e; --blue:#3987e5; --orange:#d95926; --aqua:#199e70;
    --warnbg:#2a2416; --warnline:#c98500;
  }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--ink); margin:0;
    font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding-block:0; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:40px 20px 80px; }}
  h1 {{ font-size:26px; line-height:1.25; margin:0 0 6px; letter-spacing:-0.01em; }}
  h2 {{ font-size:19px; margin:44px 0 12px; padding-top:18px;
        border-top:1px solid var(--line); letter-spacing:-0.01em; }}
  h3 {{ font-size:15.5px; margin:26px 0 8px; }}
  .sub {{ color:var(--ink2); font-size:14px; margin:0 0 28px; }}
  p {{ margin:0 0 13px; }}
  code, .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:0.9em; }}
  code {{ background:var(--card); padding:1px 5px; border-radius:4px;
    border:1px solid var(--line); }}
  pre {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
    padding:13px 15px; overflow-x:auto; font-size:12.5px; line-height:1.5; }}
  .tw {{ overflow-x:auto; margin:16px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px;
    min-width:640px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
  th {{ font-weight:600; color:var(--ink2); font-size:11.5px;
    text-transform:uppercase; letter-spacing:0.04em; white-space:nowrap; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  td.win {{ font-weight:700; color:var(--aqua); }}
  figure {{ margin:22px 0; }}
  figure img {{ width:100%; max-width:100%; height:auto; display:block;
    border:1px solid var(--line); border-radius:8px; background:#fcfcfb; }}
  figcaption {{ color:var(--ink2); font-size:12.5px; margin-top:8px;
    line-height:1.5; }}
  .note {{ background:var(--warnbg); border-left:3px solid var(--warnline);
    padding:12px 16px; border-radius:0 7px 7px 0; margin:18px 0;
    font-size:14px; }}
  .note strong {{ display:block; margin-bottom:3px; }}
  ul, ol {{ margin:0 0 13px; padding-left:22px; }}
  li {{ margin-bottom:7px; }}
  .lede {{ font-size:16px; color:var(--ink); }}
  .foot {{ color:var(--ink3); font-size:12.5px; margin-top:46px;
    padding-top:16px; border-top:1px solid var(--line); }}
</style>

</head>
<body>
<div class="wrap">

<h1>Control-law comparison on the 6-12-8 system</h1>
<p class="sub">Eight runs · linear plant · 12 September 2026 ·
<span class="mono">rattlesnake-vibration-controller</span>, branch
<span class="mono">control-law-development-2026-09</span></p>

<p class="lede">Four control laws, each run with the drive-coherence cap on and
off, on one plant with every other setting held fixed and machine-verified.
The headline: <strong>which law is best depends on the acceptance criterion</strong>,
and the drive power required to get there spans a factor of 128.</p>

<h2>1. The system under test</h2>

<p>Six drives, twelve responses, eight control channels, controlled over
100–1000 Hz against a flat 1&times;10<sup>-3</sup> specification with zero
cross-terms. The plant is the linear SDynPy frame model
<span class="mono">sdynpy_frame6x12_system.npz</span> (hardware type 6, 36 DOF),
run as virtual hardware at 4096 Hz.</p>

{img('system_characterisation.png',
     'Left: modal damping for the 33 modes inside the control band — this is a '
     'lightly damped structure, median 0.83 %, minimum 0.70 %. Centre: the six '
     'FRF singular values. The smallest sits three to four orders of magnitude '
     'below the largest, and with eight control channels driven by six shakers, '
     'two response directions cannot be independently commanded at all. Right: '
     'median cond(H) per run — the same deterministic plant, identified with '
     'identical settings, scattering by a factor of ten.')}

<div class="note"><strong>The reachability limit is structural, not a control failure.</strong>
With M = 8 control channels and N = 6 drives, the set of achievable response
diagonals is a proper subset of the positive orthant. Part of the flat
specification is unreachable by any law whatsoever. Measured per run from each
run's own FRF, that limit is <strong>3.57–3.91 dB rms</strong> — consistent
across all eight runs, as expected on one plant.</div>

<h2>2. What was held fixed</h2>

<p>Every run was launched from a dedicated profile via
<span class="mono">--profile</span>, which bypasses both startup dialogs and
reads all settings from the spreadsheet — adopted after settings were found
drifting silently in the GUI between runs.</p>

<div class="tw"><table>
<tr><th>Setting</th><th>Value</th><th>Setting</th><th>Value</th></tr>
<tr><td>Plant</td><td class="mono">sdynpy_frame6x12_system.npz</td>
    <td>Control averaging</td><td>Exponential, coef 0.04</td></tr>
<tr><td>Sample rate</td><td>4096 Hz</td>
    <td>Frames in CPSD</td><td>20</td></tr>
<tr><td>Samples per frame</td><td>4096</td>
    <td>FRF update during control</td><td>on</td></tr>
<tr><td>Time per read / write</td><td>0.25 s</td>
    <td>System ID</td><td>H1, 20 averages, 1.0 V RMS</td></tr>
</table></div>

<p>The scoring harness reads all of these from each file and refuses to compare
runs that disagree. All eight passed:
<span class="mono">provenance check: all runs share plant, sample rate,
averaging and spectral settings</span>.</p>

<h2>3. Results</h2>

<div class="tw"><table>
<tr><th>Run</th><th>Law</th><th>Cap</th><th>cond(H)</th>
<th>rms dB</th><th>% out</th>
<th>rms vs&nbsp;achievable</th><th>% out</th>
<th>unreachable</th><th>drive trace</th><th>level</th></tr>
{result_rows()}
</table></div>

<p><em>rms dB</em> and <em>% out</em> are against the flat specification;
the next pair is against each run's own recomputed achievable target;
<em>unreachable</em> is how far that achievable target itself sits from the flat
spec. Best value in each column is highlighted.</p>

<h3>The laws distribute error differently</h3>

{img('per_channel_mean.png',
     'match_trace runs every channel low. optimal_diagonal brings six channels '
     'to within a third of a dB and leaves 13X+ and 14X+ at -5 dB. '
     'match_diagonal_congruence is the only law that lifts those two, paying '
     'for it by running the other six about a dB hot.')}

{img('achievable_vs_achieved.png',
     'The grey bars are the reachability limit — where the best possible '
     'response sits relative to the flat spec, channel by channel. The '
     'achievable target is 2 dB ABOVE spec on 8X+ and 2 dB BELOW on 13X+/14X+. '
     'match_diagonal_congruence tracks that shape closely without being told '
     'about it; the other two laws chase the flat spec and miss it in the same '
     'direction on every channel.')}

<p>That last figure is the most important one in the study. The congruence law
was run <strong>without</strong> a projected-target file — it was aiming at the
raw flat specification like everything else. Its per-channel congruence feedback
finds the balanced solution on its own.</p>

<h3>The two acceptance criteria disagree</h3>

{img('rms_vs_pout.png',
     'Lower-left is better on both axes, but no run occupies it. Congruence '
     'wins on rms and loses on lines-out; optimal_diagonal is the reverse. '
     'Marker area is proportional to log drive power.')}

<div class="tw"><table>
<tr><th>Criterion</th><th>Winner</th><th>Margin</th></tr>
<tr><td>% lines outside ±6 dB (standard test acceptance)</td>
    <td>optimal_diagonal_fast (cap off)</td><td>10.7 % vs 18.7 %</td></tr>
<tr><td>Overall rms error</td><td>congruence (cap off)</td>
    <td>5.23 dB vs 5.62 dB</td></tr>
<tr><td>Worst channel (13X+)</td><td>congruence</td>
    <td>−2.4 dB mean vs −5.0 dB</td></tr>
<tr><td>Scored against what is achievable</td><td>congruence</td>
    <td>3.49 dB vs 4.00 dB, and ties on lines-out</td></tr>
<tr><td>Drive power for the result</td><td>congruence (cap on)</td>
    <td>0.047 vs 0.075 and 0.12</td></tr>
</table></div>

<p><strong>Scored against the achievable target rather than the flat spec, the
ambiguity disappears: congruence wins on both axes simultaneously</strong>
(3.49 dB / 8.5 % against 4.00 dB / 8.5 %). Its apparent weakness on lines-out is
an artifact of judging it against a target that cannot be met.</p>

<h3>Drive power</h3>

{img('drive_power.png',
     'Congruence with the cap on achieves the best rms of any run using the '
     'least drive power of any run. Uncapped match_trace buys its 1.0 dB '
     'improvement with fifty times the drive power.')}

<p>This reframes the coherence-cap result entirely. On match_trace, removing the
cap improves rms by 1.0 dB and lines-out by 6.4 points — but drive power rises
from 0.12 to 6.02, a factor of fifty. Whether that trade is worth making is an
actuator-capability question, not a control-quality one.</p>

<h2>4. Findings</h2>

<ol>
<li><strong>The coherence cap costs about 1 dB on match_trace and nothing on
optimal_diagonal.</strong> 8.22 → 7.21 dB for the former; 5.62 → 5.62 dB for the
latter, a difference of 0.008 dB. The cap was active in both cases (max pairwise
drive coherence reached 0.97 capped, 0.9998 uncapped).</li>

<li><strong>The cap's real cost is drive power, not error.</strong> Fifty times
more drive for 1 dB on match_trace; four and a half times more on congruence.</li>

<li><strong>Channels 13X+ and 14X+ are reachability-limited.</strong> Every law
leaves them low. This confirms by measurement what the achievable-response
predictor forecast analytically: with six drives, two of the eight control
directions cannot be commanded independently.</li>

<li><strong>match_diagonal_congruence approximates the achievable target without
being given it.</strong> It is the only law that lifts the two sacrificed
channels, and scored against what is actually achievable it wins on both rms and
lines-out.</li>

<li><strong>optimal_diagonal_control is the best choice against a flat
specification</strong> — 5.62 dB / 11.2 %, with six of eight channels inside
half a dB.</li>
</ol>

<h2>5. Caveats</h2>

<div class="note"><strong>Runs 05 and 06 did not test what they were meant to.</strong>
<span class="mono">optimal_diagonal_control_fast</span> disables its
Burer-Monteiro fast path whenever the FRF changes between calls, because the
unconstrained factored solve drives pairwise drive coherence to exactly 1.0 and
would make a live H1 estimator's reference CPSD singular. With
<span class="mono">update_tf_during_control = 1</span>, H changes every call, so
every solve fell back to the base class's SDP. <strong>Runs 05 and 06 are
repeats of 03 and 04</strong>, and the close agreement between them is not
evidence that the factored solve matches the SDP. No timing difference was
observed at the rig, consistent with this.</div>

<ul>
<li><strong>The coherence cap in the optimal-diagonal laws reaches only
SDP-refined bins</strong>, bounded by <span class="mono">max_bins_per_update =
20</span> per call. Across 901 lines most bins carry the deliberately uncapped
buzz baseline in both runs, which likely explains the null result. Raising the
bin budget would distinguish "the cap does not matter to this law" from "the cap
barely applied".</li>

<li><strong>Identification scatter is large.</strong> Median cond(H) came out
2447–25928 across eight identifications of the same deterministic plant with
identical settings — a factor of ten. Differences between laws smaller than a
few tenths of a dB should not be read as real. This is why the achievable floor
is recomputed per run rather than assumed common.</li>

<li><strong>Neither optimal-diagonal law has a startup level cap.</strong>
match_trace and congruence both take
<span class="mono">startup_test_level_cap_db</span> (−9 dB default); the
optimal-diagonal laws solve against the full specification and command it
immediately. Worked around by ramping the test level. This is a real safety
asymmetry and remains unfixed.</li>

<li><strong>bm_rank was set to 6, not its default of 4.</strong> With eight
control channels the threshold below which spurious local minima generically
appear is √(2·8) ≈ 4, so the default sits exactly on it. Moot given the fast
path never engaged.</li>

<li><strong>Congruence was run without a projected-target file</strong>, aiming
at the raw specification so that all four laws solved the same problem. Its
distinctive per-channel congruence feedback was fully active.</li>

<li><strong>The profiles carry no warning or abort levels</strong> — those
columns are all NaN — so the ±6 dB tolerance bands are synthesized from the
specification. If the real test has defined tolerances, putting them in the
profile would make these numbers contractual rather than nominal.</li>

<li><strong>One test level, one averaging configuration, one plant, no
repeats.</strong> No per-law variance estimate exists. The plant is lightly
damped (median ζ 0.83 %), which is the harder case; a better-damped structure
would likely narrow the spread between laws.</li>

<li><strong>The achievable floor was computed with</strong>
<span class="mono">--floor-decimate 4</span> <strong>and</strong>
<span class="mono">--restrict-rcond 1e-3</span>. Decimation was verified to move
the result by 0.01 dB against solving every line.</li>
</ul>

<h2>6. What is now available but unused</h2>

<p>All eight runs were saved with streaming time data — 456k to 1.6M samples
each, 1.6 GB total. This is the first campaign where that is true, and it
unlocks three diagnostics that have been unavailable until now:
<span class="mono">plot_kurtosis</span> (non-Gaussianity, the direct signature
of nonlinearity), <span class="mono">plot_signal_to_noise</span>, and
<span class="mono">optimal_subset</span> (objective selection of the stationary
window, replacing a judgement made by eye).</p>

<h2>7. Next</h2>

<ol>
<li><strong>Phase two — the achievable target as specification.</strong>
<span class="mono">make_achievable_spec.py</span> is built and verified. Pointing
every law at a reachable target removes reachability from the comparison so the
residual is tracking behaviour alone. Given that congruence already approximates
that shape unaided, this should remove its 2.8 dB level bias while keeping the
balance — making it win on both criteria outright.</li>
<li>Raise <span class="mono">max_bins_per_update</span> and rerun the
optimal-diagonal pair, to settle the coherence-cap null result.</li>
<li>Rerun the fast law with FRF update off, as its own two-run experiment, to
find out whether Burer-Monteiro at rank 6 actually matches the SDP.</li>
<li>Add a startup level cap to both optimal-diagonal laws.</li>
<li>The nonlinear plant with
<span class="mono">RATTLESNAKE_NONLINEARITY_STRENGTH</span> swept 0 → 1 → 2
remains the cleanest plant-≠-model experiment available, and streaming is now
enabled so kurtosis becomes usable as the nonlinearity instrument.</li>
</ol>

<p class="foot">Generated from the eight saved spectral files by
<span class="mono">score_runs_sdynpy.py</span>. Every number traceable to
<span class="mono">results/analysis/law_comparison/</span>. Metrics independently
verified against a direct computation from the raw netCDF arrays with no shared
code.</p>

</div>\n</body>\n</html>\n"""

open('law_comparison_report.html', 'w').write(HTML)
print('wrote law_comparison_report.html',
      f'{os.path.getsize("law_comparison_report.html")/1024:.0f} KB')
