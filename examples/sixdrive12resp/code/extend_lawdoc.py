"""Add pseudoinverse_control and buzz_control to the control-law reference.

The 2026-09-13 reference covered five subjects and predates runs 09-12. These
two laws are now four of the twelve runs in the comparison report, and one of
them (buzz) produced the most surprising result in the campaign, so the
reference was materially incomplete.

Splices two new sections in ahead of the lineage section rather than
regenerating the document: the original generator lived in a scratch directory
that no longer exists, and the surviving HTML is the authoritative copy.
"""
import base64
import os
import re

SRC = ('/mnt/user-data/uploads/nhunterjr--Code--python--rattlesnake-vibration-'
       'controller/Claude outputs/control_law_reference.html')
OUT = 'control_law_reference.html'


def img(path, alt):
    b = base64.b64encode(open(path, 'rb').read()).decode()
    return (f'<figure class="flow"><img style="max-width:560px" '
            f'src="data:image/png;base64,{b}" alt="{alt}">'
            f'<figcaption>{alt}</figcaption></figure>')


PSEUDO = """<h2>5. pseudoinverse_control <span class="tag">function · control_laws.py</span></h2>

<p>The simplest law in the set and the oldest: one open-loop pseudoinverse solve
against the raw specification, recomputed identically every cycle. No error
feedback, no trace management, no startup guard. It is the baseline against
which every other law's extra machinery earns its place.</p>

<h3>Algorithm</h3>
<div class="eq">
<span class="c">every call, identically</span><br>
H<sup>+</sup> = pinv(H, rcond)<br>
X = H<sup>+</sup> G<sub>xx</sub> (H<sup>+</sup>)<sup>H</sup><br>
X ← cap(X, max_drive_coherence)
</div>

<p>That is the entire law. <span class="mono">last_response_cpsd</span> appears
in the signature and in the docstring — labelled <em>"Last Control Response for
Error Correction"</em> — and is <strong>never referenced in the body</strong>.
<span class="mono">last_output_cpsd</span> is not read either. There is no
integrator and no error term of any kind, so the law cannot converge toward the
specification; it lands wherever the current FRF estimate puts it and stays
there.</p>

<p>The only thing that changes between cycles is <span class="mono">H</span>.
With <span class="mono">update_tf_during_control = Y</span> the FRF is
re-identified from live control data, so the law does adapt — but to a changing
plant model, not to its own error. That distinction is the difference between an
adaptive feedforward controller and a feedback one.</p>

<div class="note"><strong>Why it is expensive</strong>
The flat specification is diagonal, hence rank 8, while
<span class="mono">H X H<sup>H</sup></span> has rank at most 5 on this plant.
Inverting a maximally unreachable target with no trace or coherence management
burns drive power trying to manufacture response components that do not exist in
the column space. Measured: 5.38&nbsp;V² uncapped, roughly seventy times the
optimal-diagonal family, for a worse score.</div>

{FLOW_PSEUDO}

<h3>Parameters</h3>
<div class="tw"><table>
<tr><th>Position</th><th>Name</th><th>Default</th><th>Meaning</th></tr>
<tr><td>1</td><td class="mono">rcond</td><td>1e-15</td>
    <td>Pseudoinverse conditioning cutoff, applied on every call. 1e-3 was used in this study; on this plant that truncation is correct, since the sixth singular direction does not physically exist.</td></tr>
<tr><td>2</td><td class="mono">max_drive_coherence</td><td>1.0</td>
    <td>Cap on pairwise drive coherence. <strong>1.0 is a true no-op.</strong> Parsed by <span class="mono">_parse_rcond_and_cap</span>; there is no third parameter.</td></tr>
</table></div>

<div class="note"><strong>Open defect: no startup guard</strong>
This is the only law in the set without the first-cycle clamp added 2026-09-02.
On frame one the FRF is a single unaveraged estimate under both averaging modes,
and a poorly conditioned first-frame solve can command a level decades above
anything the loop would settle at. match_trace, buzz and congruence all take
<span class="mono">startup_test_level_cap_db</span>; this one does not.</div>

<h3>Measured</h3>
<div class="tw"><table>
<tr><th>Run</th><th>Cap</th><th>rms dB</th><th>% out</th><th>vs floor</th><th>drive V²</th><th>level dB</th></tr>
<tr><td class="mono">09</td><td>0.95</td><td>10.47</td><td>42.1</td><td>9.55</td><td>5.449</td><td>+13.55</td></tr>
<tr><td class="mono">10</td><td>off</td><td>7.01</td><td>17.8</td><td>5.48</td><td>5.377</td><td>−1.28</td></tr>
</table></div>

<p>Worst in the twelve-run matrix on both acceptance criteria with the cap on.
The cap costs it <strong>3.47&nbsp;dB</strong> — see section 7 for why the cap
inverts its meaning for laws without feedback.</p>
"""

BUZZ = """<h2>6. buzz_control <span class="tag">function · control_laws.py</span></h2>

<p>pseudoinverse_control with one substitution, and that substitution is worth
more than every feedback mechanism in the rest of the document. Before
inverting, it replaces the specification's coherence and phase with the
<em>measured</em> coherence and phase from the system-ID buzz test, keeping the
specification's own auto-spectra.</p>

<h3>Algorithm</h3>
<div class="eq">
<span class="c">the substitution — match_coherence_phase(G<sub>xx</sub>, G<sub>buzz</sub>)</span><br>
γ²<sub>mn</sub> = |G<sub>buzz,mn</sub>|² / (G<sub>buzz,mm</sub> G<sub>buzz,nn</sub>)&nbsp;&nbsp;<span class="c">measured coherence</span><br>
φ<sub>mn</sub> = ∠G<sub>buzz,mn</sub>&nbsp;&nbsp;<span class="c">measured phase</span><br>
a<sub>m</sub> = G<sub>xx,mm</sub>&nbsp;&nbsp;<span class="c">specified auto-spectra</span><br>
S′<sub>mn</sub> = e<sup>iφ<sub>mn</sub></sup> √(γ²<sub>mn</sub> a<sub>m</sub> a<sub>n</sub>)<br><br>
<span class="c">then the same open-loop solve</span><br>
X = H<sup>+</sup> S′ (H<sup>+</sup>)<sup>H</sup><br>
<span class="c">first call only</span><br>
X ← min(1, 10<sup>cap/10</sup> tr(G<sub>xx</sub>) / tr(H X H<sup>H</sup>)) · X<br>
X ← cap(X, max_drive_coherence)
</div>

<p><strong>Why the substitution matters.</strong> The specification's implied
coherence is zero everywhere — eight mutually uncorrelated responses — which a
rank-5 plant cannot produce at all. The measured coherence, by construction,
<em>is</em> something the plant produces. So buzz inverts a target that lies
close to the achievable set while pseudoinverse inverts one that lies far
outside it. Same solver, same conditioning, same cap: 1.1&nbsp;dB better and a
third of the drive.</p>

<p>Like pseudoinverse_control, it has <strong>no error feedback</strong>:
<span class="mono">last_response_cpsd</span> is accepted and never read, and
<span class="mono">last_output_cpsd</span> is used only as a sentinel to fire the
startup clamp once. It also has a second, less obvious limitation — the two
inputs refresh on different schedules.</p>

<div class="note"><strong>The buzz target is frozen; only the FRF adapts</strong>
In <span class="mono">random_vibration_sys_id_data_analysis.py</span>,
<span class="mono">update_tf_during_control</span> swaps in the re-identified
<span class="mono">control_frf</span>. But
<span class="mono">sysid_response_cpsd</span> — the source of the coherence and
phase being matched — is assigned only in
<span class="mono">run_sysid_transfer_function</span> and is never refreshed
during control. The target shape is fixed for the whole run.</div>

{FLOW_BUZZ}

<h3>Parameters</h3>
<div class="tw"><table>
<tr><th>Position</th><th>Name</th><th>Default</th><th>Meaning</th></tr>
<tr><td>1</td><td class="mono">rcond</td><td>1e-15</td>
    <td>Pseudoinverse conditioning cutoff, applied on every call. 1e-3 was used in this study.</td></tr>
<tr><td>2</td><td class="mono">max_drive_coherence</td><td>1.0</td>
    <td>Cap on pairwise drive coherence. <strong>1.0 is a true no-op.</strong></td></tr>
<tr><td>3</td><td class="mono">startup_test_level_cap_db</td><td>−9.0</td>
    <td>Ceiling on the first command only. Unlike match_trace there is no separate steady-state branch to guard, so every later call is completely unaffected.</td></tr>
</table></div>

<p>Shares <span class="mono">_parse_match_trace_parameters</span> with
match_trace_pseudoinverse, so the parameter string is identical in form.</p>

<h3>Measured</h3>
<div class="tw"><table>
<tr><th>Run</th><th>Cap</th><th>rms dB</th><th>% out</th><th>vs floor</th><th>drive V²</th><th>level dB</th></tr>
<tr><td class="mono">11</td><td>0.95</td><td>7.90</td><td>30.2</td><td>6.75</td><td>0.981</td><td>+9.38</td></tr>
<tr><td class="mono">12</td><td>off</td><td>5.91</td><td>13.6</td><td>4.20</td><td>1.473</td><td>−0.30</td></tr>
</table></div>

<p>Uncapped, buzz_control finishes <strong>0.68&nbsp;dB behind the best run in
the entire twelve-run matrix</strong> and ahead of both match_trace runs — with
no error feedback whatsoever. On this plant the 5.2–5.9&nbsp;dB cluster is set by
reachability and identification quality, not by loop design; what separates the
laws is how sensibly each picks a target inside the reachable set.</p>

<h3>The generator and class forms are not equivalent</h3>

<p><span class="mono">control_laws.py</span> also contains
<span class="mono">buzz_control_generator</span> and
<span class="mono">buzz_control_class</span>. Neither has the drive-coherence
cap, neither takes <span class="mono">rcond</span>, and neither has the startup
guard — they call a bare <span class="mono">np.linalg.pinv</span>. The generator
caches <span class="mono">modified_spec</span> on its first call and never
recomputes it. <span class="mono">buzz_control_class.control()</span> accepts
<span class="mono">last_response_cpsd</span> and never references it, replaying
one open-loop solve forever; its specification updates only through
<span class="mono">system_id_update</span>. Both were excluded from the
comparison deliberately.</p>
"""

CAP_SECTION = """<h2>7. The coherence cap means opposite things to the two families</h2>

<p><span class="mono">_cap_drive_coherence</span> shrinks any off-diagonal drive
term whose pairwise coherence exceeds the cap, preserves every diagonal value,
then re-projects onto the PSD cone by clipping negative eigenvalues. Applied to
a <em>feedback</em> law it acts as a constraint the loop re-converges around.
Applied to an <em>open-loop</em> law it is pure damage, because there is nothing
to re-converge with.</p>

<div class="tw"><table>
<tr><th>Law</th><th>Loop</th><th>rms cap on</th><th>rms cap off</th><th>Δ</th><th>level on</th><th>level off</th></tr>
<tr><td class="mono">match_diagonal_congruence</td><td>feedback</td><td>5.31</td><td>5.23</td><td>−0.08</td><td>+2.85</td><td>+2.78</td></tr>
<tr><td class="mono">optimal_diagonal_control</td><td>feedback</td><td>5.62</td><td>5.62</td><td>−0.01</td><td>+0.08</td><td>+0.41</td></tr>
<tr><td class="mono">optimal_diagonal_control_fast</td><td>feedback</td><td>5.64</td><td>5.66</td><td>+0.02</td><td>+0.26</td><td>+0.06</td></tr>
<tr><td class="mono">match_trace_pseudoinverse</td><td>feedback</td><td>8.22</td><td>7.21</td><td>−1.01</td><td>+0.09</td><td>+0.05</td></tr>
<tr><td class="mono">buzz_control</td><td>open</td><td>7.90</td><td>5.91</td><td>−1.99</td><td>+9.38</td><td>−0.30</td></tr>
<tr><td class="mono">pseudoinverse_control</td><td>open</td><td>10.47</td><td>7.01</td><td>−3.47</td><td>+13.55</td><td>−1.28</td></tr>
</table></div>

<p><strong>The mechanism.</strong> Both open-loop solves are strongly
rank-concentrated — eigenvalue participation ratio 1.0–1.1 out of 6 — so the six
shakers act as essentially one source and the responses cancel by precise
inter-drive phasing. That phasing lives entirely in the off-diagonal terms. The
cap shrinks them while preserving the diagonal, so drive power is barely
changed but the cancellation is destroyed and responses add incoherently: the
level jumps roughly 20&nbsp;dB. The residual is <em>shape</em> error, not level
— the best possible single rescale of run 09's entire drive recovers only
0.30&nbsp;dB.</p>

<div class="note"><strong>Operational consequence</strong>
For <span class="mono">pseudoinverse_control</span> and
<span class="mono">buzz_control</span>, <strong>cap off (1.0) is the correct
setting</strong>, not the reckless one. This is the reverse of the feedback-law
intuition and nothing in the parameter name or the docstrings warns of it.
Confirmed in reverse: the best achievable drive found by the floor solver sits
above 0.95 coherence on 49&nbsp;% of its pairs, so the cap constrains the
solution away from the optimum by construction.</div>
"""


def main():
    s = open(SRC).read()

    body = (PSEUDO.replace('{FLOW_PSEUDO}',
                           img('flowcharts/06_pseudoinverse.png',
                               'pseudoinverse_control. One open-loop solve, '
                               'recomputed identically every cycle.'))
            + '\n\n' + BUZZ.replace('{FLOW_BUZZ}',
                                    img('flowcharts/07_buzz.png',
                                        'buzz_control. The substitution happens '
                                        'before the solve; everything after it '
                                        'is pseudoinverse_control.'))
            + '\n\n' + CAP_SECTION + '\n\n')

    # renumber the two sections that follow, back to front so the indices hold
    s = s.replace('<h2>6. Lineage and measured results</h2>',
                  '<h2>8. Lineage and measured results</h2>')
    s = s.replace('<h2>5. achievable_response', '<h2>MOVED_ACH')
    anchor = '<h2>MOVED_ACH'
    i = s.index(anchor)
    s = s[:i] + body + s[i:]
    s = s.replace('<h2>MOVED_ACH', '<h2>9. achievable_response')

    # achievable_response reads better last, after the laws it measures; move it
    j = s.index('<h2>9. achievable_response')
    k = s.index('<h2>8. Lineage and measured results</h2>')
    ach, rest = s[j:k], s[k:]
    s = s[:j] + rest.replace('<h2>8. Lineage', '<h2>8. Lineage', 1)
    s = s.replace('<h2>8. Lineage and measured results</h2>',
                  ach.replace('<h2>9. achievable_response',
                              '<h2>8. achievable_response')
                  + '<h2>9. Lineage and measured results</h2>', 1)

    s = s.replace(
        'Five subjects: the four control laws compared on hardware, plus the\n'
        'achievable-response predictor that supplies the denominator every comparison is\n'
        'measured against. Each section gives the algorithm, the governing equations, a\n'
        'flow chart, the parameter string, and what was actually measured.',
        'Seven subjects: the six control laws compared on hardware, plus the '
        'achievable-response predictor that supplies the denominator every '
        'comparison is measured against. Each section gives the algorithm, the '
        'governing equations, a flow chart, the parameter string, and what was '
        'actually measured. Sections 5 and 6 cover the two original upstream '
        'open-loop laws, added 15 September 2026 after runs 09–12; section 7 '
        'covers the drive-coherence cap, whose meaning turns out to depend on '
        'which family it is applied to.')
    s = s.replace('13 September 2026 · <span class="mono">control_laws/</span>',
                  '15 September 2026 · <span class="mono">control_laws/</span>')

    open(OUT, 'w').write(s)
    heads = re.findall(r'<h2>([^<]{0,60})', s)
    print(f'wrote {OUT}  {os.path.getsize(OUT)/1024:.0f} KB')
    for h in heads:
        print('   ', h)


if __name__ == '__main__':
    main()


def update_lineage():
    """Bring the lineage table up to twelve runs and six laws."""
    s = open(OUT).read()
    s = s.replace(
        '<tr><td class="mono">match_diagonal_congruence</td><td>structural, D X D</td>'
        '<td>N</td><td>5.31</td><td>19.8</td><td>0.047</td></tr>\n</table></div>',
        '<tr><td class="mono">match_diagonal_congruence</td><td>structural, D X D</td>'
        '<td>N</td><td>5.31</td><td>19.8</td><td>0.047</td></tr>\n'
        '<tr><td class="mono">pseudoinverse_control</td><td>none needed — X is a '
        'congruence of G<sub>xx</sub> ⪰ 0</td><td>0 (open loop)</td><td>10.47</td>'
        '<td>42.1</td><td>5.449</td></tr>\n'
        '<tr><td class="mono">buzz_control</td><td>none needed — X is a congruence '
        'of S′ ⪰ 0</td><td>0 (open loop)</td><td>7.90</td><td>30.2</td>'
        '<td>0.981</td></tr>\n</table></div>')
    s = s.replace(
        '<p>Results are the cap-on runs from the 12 September comparison, linear plant,\n'
        'against the flat specification. Scored instead against what is\n'
        '<em>achievable</em>, congruence leads at 3.49 dB against optimal_diagonal\'s\n'
        '4.00 dB.</p>',
        '<p>Results are the <strong>cap-on</strong> runs from the 15 September '
        'twelve-run comparison, linear plant, against the flat specification. '
        'Cap-on is the wrong setting for the two open-loop laws (section 7), so '
        'their rows understate them badly: uncapped they measure 7.01 and 5.91 dB '
        'with 17.8 % and 13.6 % of lines out. Scored instead against what is '
        '<em>achievable</em>, congruence leads at 3.48 dB, the optimal-diagonal '
        'family sits at a flat 3.99, buzz at 4.20, pseudoinverse at 5.48 and '
        'match_trace at 5.78.</p>\n\n'
        '<p>The two open-loop laws need no positive-semidefiniteness machinery at '
        'all: X = H<sup>+</sup> S (H<sup>+</sup>)<sup>H</sup> is a congruence '
        'transform of a PSD matrix, so validity is automatic. Every guarantee in '
        'the table above buys the ability to <em>shape</em> the drive, not the '
        'ability to keep it valid — and on this plant that shaping is worth '
        'between 0 and 1.1 dB against a well-chosen target, which is the '
        'campaign\'s least comfortable result.</p>')
    s = s.replace(
        'law\'s central claim, that Burer-Monteiro at rank 6 matches the SDP, remains\n'
        'untested on this system because FRF update disabled its fast path.</div>',
        'law\'s central claim, that Burer-Monteiro at rank 6 matches the SDP, remains\n'
        'untested on this system because FRF update disabled its fast path. '
        '<span class="mono">pseudoinverse_control</span> has no startup guard '
        'either, and neither <span class="mono">buzz_control_generator</span> nor '
        '<span class="mono">buzz_control_class</span> has the coherence cap or the '
        'startup guard.</div>')
    open(OUT, 'w').write(s)
    print('lineage updated')


update_lineage()
