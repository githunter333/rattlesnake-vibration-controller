# -*- coding: utf-8 -*-
"""Build results/notes/all_runs_chart.txt -- every run 01-56, one block each.

Numbers are read from report_data.json; descriptions come from the runsheets
and session summaries. Assembled from _chart_part*.py by make_run_chart.py.
"""
import json, os, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.normpath(os.path.join(HERE, '..', 'results', 'notes'))
OUT = os.path.join(NOTES, 'all_runs_chart.txt')
LIN = {r['run']: r for r in json.load(open(os.path.join(HERE, 'report_data.json')))}

# run: (plant, law, config, description)
R = {}
R[1] = ('LIN','match_trace_pinv','cap 0.95',
 'Cap-ON half of the reference pair for this law. Worst of the four laws carried '
 'forward, and the weakness is structural rather than stale code -- run 33 reproduced '
 'it four code generations later.')
R[2] = ('LIN','match_trace_pinv','cap off',
 'Cap-OFF partner of run 01. Prices the drive-coherence cap on this law at 1.01 dB: '
 'match_trace has only a single real scalar per line left to repair the damage after '
 'the cap is post-processed onto an already-computed solve.')
R[3] = ('LIN','optimal_diagonal','cap 0.95',
 "Cap-ON point for the SDP law and the project's accuracy benchmark. Reproduced at "
 'runs 19 and 34 across three code generations, drive within 0.4% each time.')
R[4] = ('LIN','optimal_diagonal','cap off',
 'Cap-OFF partner of run 03. Identical within 0.01 dB -- the cap is free here because '
 'it enters the SDP as a constraint, so the returned solution already honours it.')
R[5] = ('LIN','optimal_diag_fast','cap 0.95',
 'Intended cap-ON point for the fast variant. Later found NOT to have tested the fast '
 'path at all: the bit-equality demotion gate sent every bin to the base solver, so '
 'runs 05 and 06 are the base law under another name.')
R[6] = ('LIN','optimal_diag_fast','cap off',
 'Cap-OFF partner of run 05, with the same gate defect.')
R[7] = ('LIN','congruence','cap 0.95',
 'Cap-ON point for the congruence law. Good rms but +2.85 dB of level error. This law '
 'has never held total level on any run in the programme.')
R[8] = ('LIN','congruence','cap off',
 'Cap-OFF partner. Best raw accuracy of any law in the set at 5.23 dB, and still 2.79 dB '
 'off level. The cap costs only 0.08 dB because the law re-converges per drive around it.')
R[9] = ('LIN','pseudoinverse','cap 0.95',
 'Opens the open-loop family: no feedback and no error integrator. The cap is pure damage '
 'here because there is no mechanism downstream to repair what it breaks.')
R[10] = ('LIN','pseudoinverse','cap off',
 'Cap-OFF partner. The 3.47 dB gap to run 09 is the largest cap cost in the set and makes '
 'cap-off the operationally correct setting for this family -- the reverse of the usual '
 'intuition about the cap.')
R[11] = ('LIN','buzz','cap 0.95',
 'Cap-ON point for the second open-loop law. The cap costs 1.99 dB.')
R[12] = ('LIN','buzz','cap off',
 "The runsheet's headline prediction was about 3.9 dB, beating every feedback law. It did "
 "not: 5.91 dB against optimal_diagonal's 5.62.")
R[13] = ('LIN','optimal_diag_fast','update off',
 'INADMISSIBLE. Saved metadata reads control law = pseudoinverse_control with update on, so '
 'profile_13 was never loaded. The earlier "weak system ID" explanation is WITHDRAWN -- '
 'scored like for like, this run had the best identification of the set.')
R[14] = ('LIN','optimal_diag_fast','update off',
 'Redo of run 13 with the profile actually loaded. Identification accepted at median 0.998, '
 'then stopped when it appeared hung. No spectral file was saved, so the linear-objective '
 'baseline it was meant to establish was never measured.')
R[15] = ('LIN','optimal_diag_fast','dB objective, rcond 1e-2',
 'Swaps the linear-units squared-error objective for a dB one so the solver stops trading '
 'good channels against hard ones. Error was not broadband: a structural null at 418-428 Hz '
 'plus a self-inflicted band-edge failure at 985-1000 Hz.')
R[16] = ('LIN','optimal_diagonal','dB objective via IRLS',
 'The same dB objective reached by a different solver, to be read against run 15. Its '
 'spectral-file FRF differed from its own system ID by 31.6%, matching run 15 at 31.7% -- '
 'two independent runs agreeing to a tenth of a percent.')
R[17] = ('--','optimal_diag_fast','dB obj, rcond 3e-3',
 'NOT RUN. Planned drive_rcond A/B to recover the 985-1000 Hz failure. Prepared -- the '
 'superseded profile is still in the profiles folder -- then overtaken when the dB objective '
 'was defaulted back off on 09-18.')
R[18] = ('--','optimal_diag_fast','dB obj, update on',
 'NOT RUN. Planned update-on A/B at fixed drive_rcond. Same fate as run 17.')
R[19] = ('LIN','optimal_diagonal','cap 0.95, profile_03 unedited',
 'Re-run of the benchmark on current code, the only intended difference being the fixed '
 'startup level cap. Reproduced run 03.')
R[20] = ('LIN','congruence','cap 0.95, profile_07 unedited',
 'Same refresh for congruence. A low rms with a large fraction of lines out at +3.04 dB '
 'means the errors are balanced, not small.')
R[21] = ('LIN','optimal_diag_fast','cap 0.95, gate fixed',
 'The one of runs 19-21 that was not a repeat: the bit-equality gate replaced by a '
 'relative-Frobenius threshold so the factored path could actually engage. Its sysid file '
 'saved as un21_ with the leading r missing.')
R[22] = ('LIN','optimal_diag_fast','drive_rcond OFF',
 'The unconstrained factored solve, deliberately the failure case. +14.3 dB level and 53% of '
 'lines out confirmed the offline prediction that a 1% drive perturbation costs +16.7 dB. The '
 'solution was unrealisable, not wrong.')
R[23] = ('LIN','optimal_diag_fast','drive_rcond 1e-2',
 'First point of the drive_rcond sweep, after the fix that made the parameter apply in the '
 'linear error domain as well as the dB one.')
R[24] = ('LIN','optimal_diag_fast','drive_rcond 3e-2',
 'Sweep continues. This is the minimum.')
R[25] = ('LIN','optimal_diag_fast','drive_rcond 5e-2',
 'Past the minimum, bracketing it at 3e-2.')
R[26] = ('LIN','optimal_diag_fast','automatic drive_rcond',
 "drive_rcond re-expressed as a multiple of the plant's own singular-value spread instead of a "
 'hand-tuned absolute ratio. The law derived 0.0302 unaided and matched hand-tuned run 24 on '
 'accuracy.')
R[27] = ('LIN','optimal_diag_fast','automatic drive_rcond, repeat',
 'Repeat of 26: derived 0.0303, accuracy repeated to 0.1 dB. Drive did not -- 153, 485 and '
 '245 V^2 across runs 24, 26 and 27. Accuracy repeats and drive does not; that finding still '
 'stands.')
R[28] = ('LIN','optimal_diag_fast','update ON, gate probe',
 "Probe of the demotion gate's own statistic. 99% of 549 calls demoted, so the fast law was "
 'byte-for-byte the base class whenever the update was on. Cause: the gate measured FRF change '
 'over all 2049 bins, 1148 of them out of band and pure noise. No stream saved.')
R[29] = ('LIN','optimal_diag_fast','update on, in-band gate, thr 0.5',
 'First run with the gate measured in band only. Drive 17.0 V^2 against a 16.2 V^2 offline '
 'prediction, a factor of nine below the bottom of the earlier 153-485 V^2 spread.')
R[30] = ('LIN','optimal_diag_fast','update on, repeat of 29',
 'The live H1 estimate DIVERGED: drift median 39834, 1.8e5 V commanded, 176 solver failures, '
 'then an absorbing dead state. Killed by the operator. The trigger is still unexplained; the '
 'system ID was checked and was normal.')
R[31] = ('LIN','optimal_diag_fast','update on, two live-FRF guards',
 'Retake of run 30 with the per-bin divergence guard and the deadlock escape in place. Clean, '
 'and 18.44 V^2 lands inside the pre-registered 15-20 V^2 confirming band. Drive decayed '
 'monotonically from 69.2 to 2.13 V.')
R[32] = ('LIN','optimal_diag_fast','update on, third repeat',
 'Third confirmation: 16.22 V^2 against the 16.20 V^2 prediction, which settled the question. '
 'Guard fired 3 times in 281 calls, all in the startup transient, with 0 solver failures. '
 'First run with the full console captured.')
R[33] = ('LIN','match_trace_pinv','cap 0.95, profile_01 unedited',
 'Refresh of the stalest row on the board. Reproduced run 01 to a tenth of a dB, so 8.22 was '
 'never a stale-code artifact: the law applies one real scalar per line and nothing in it '
 'controls how error is distributed across channels.')
R[34] = ('LIN','optimal_diagonal','cap 0.95, profile_03 unedited',
 'Third point on the benchmark across three code generations, every column unchanged. Also the '
 'cheap field test of the two live-FRF guards on a law other than the fast one; one fired on '
 'call 7, then silence for 354 calls.')
R[35] = ('LIN','optimal_diag_fast','trim OFF, profile_26 unedited',
 'Regression control for the trim pair: proves the new level-change code path does no harm and '
 'the trim is provably asleep at default. Measured model mismatch at 0.060 dB rms over 7208 '
 "channel-bins, so the residual 6.0 dB is the law's own rank-5 limit, not a model-to-rig gap.")
R[36] = ('--','optimal_diag_fast','trim ON',
 'NOT RUN. The actual trim test. Its pre-registered 0.3 dB level improvement was withdrawn once '
 'run 35 measured level at -0.023 dB and mismatch at 0.060 dB -- there was nothing left on this '
 'plant for a trim to correct.')
R[37] = ('--','optimal_diagonal','trim OFF',
 'NOT RUN. Second regression control; the runsheet itself nominated it as the one to drop if '
 'time ran short.')
R[38] = ('--','optimal_diagonal','trim ON',
 'NOT RUN. Trim test on the base SDP law. Still outstanding.')
R[39] = ('LIN','match_trace_pinv','refresh OFF, frf_diag ON',
 'Pure instrumentation, command bit-identical to run 33, to measure the per-line FRF shape '
 'change the new refresh gate would act on. Since-cycle-1 settled at 0.0522 against a 0.0070 '
 'noise floor and was DECAYING -- a step, not drift. Set the 0.05 threshold, and falsified the '
 "author's prediction that there would be nothing to track.")
R[40] = ('LIN','match_trace_pinv','gated refresh ON at 0.05',
 'The best this law has ever scored, well outside the +/-0.04 dB cluster of runs 01, 33 and 39. '
 'Refresh count decayed from 901 lines to 6-18, so run 41 was not needed. CORRECTED: the '
 'pre-registered order-of-magnitude drive DROP was falsified -- drive rose 20.6%. The refreshed '
 'shape uses all six drives instead of two nearly cancelling ones; it buys accuracy by paying drive.')
R[41] = ('--','match_trace_pinv','refresh threshold 0.10',
 'NOT RUN. Contingency only, to be taken if run 40 showed continuous re-solving or instability. '
 'Run 40 was clean, so the 39/40 pair is the result.')
R[42] = ('NL-c1','match_trace_pinv','cubic, 1 mode, strength 1',
 'First run on any article other than the linear plant. STOPPED AFTER SYSTEM ID FOR LACK OF '
 'EFFECT, not escape: q_rms/q0_ref = 0.0067, so even strength 400 would give only -0.018% shift. '
 'Its console log is six lines with zero overflow warnings.')
R[43] = ('--','match_trace_pinv','cubic 1 mode, refresh ON',
 'NOT RUN. The refresh partner of run 42, superseded once run 42 showed the plant did not move.')
R[44] = ('NL-call','match_trace_pinv','cubic, all 33 modes, strength 100',
 'ESCAPE during system identification: k3s*q**3 overflowed with k3 reaching -2.5e19 and NaNs '
 'went into the FFT convolution. This is passing q_esc = wn/sqrt|k3|, not a Duffing jump. '
 'Strength 25 would have failed too. System ID kept, stream unreadable.')
R[45] = ('--','match_trace_pinv','all-modes cubic, refresh ON',
 'NOT RUN. Abandoned after run 44 escaped.')
R[46] = ('NL-cesc','match_trace_pinv','escape-calibrated cubic, margin 10',
 'ESCAPE TWICE. The first build of the plant at margin 6 overflowed inside system ID; the final '
 'plant escaped at control cycle 3 at -18 dB. The -9 dB startup cap holds cycle 1 low and an '
 'unclamped unity-gain integrator then recovers all 9 dB in a single step. Stream unreadable.')
R[47] = ('--','match_trace_pinv','escape-cal cubic, refresh ON',
 'NOT RUN. Abandoned after run 46 escaped.')
R[48] = ('NL-cesc','match_trace_pinv','gap rank selection (rcond -0.05)',
 'ESCAPE, but much later: clean through system ID and 175 cycles at -18 and -12 dB, then escaped '
 'on the step to 0 dB. Established that the distortion-inflated singular spectrum is a SYSTEM-ID '
 'artifact and the control-time estimate is essentially the linear plant. 83.6 MB of stream lost '
 'to pkill -9.')
R[49] = ('NL-bs','match_trace_pinv','bounded softening, refresh OFF',
 'Fifth attempt at the same experiment, on the kernel that structurally cannot escape. Ran to '
 '+8 dB and 30.6 V rms with ZERO overflow across 498 cycles and 8 spectra. The plant moved only '
 'where it was built to: the six linear modes stayed at estimator noise while the 27 softened '
 'ones rose 18x. The apparent rms improvement with level was an ARTIFACT -- distortion scored as '
 'signal; on the coherent column control is flat to slightly worse.')
R[50] = ('NL-bs','match_trace_pinv','bounded softening, refresh ON 0.05',
 'One parameter different from run 49. Better on the coherent column at every level, by 0.52, '
 '0.55, 0.62 and 0.63 dB. The linear control group split the gate\'s two jobs: 0.62 dB below '
 '300 Hz is estimate refinement, 1.5-2.6 dB above it is genuine plant tracking. Two surprises -- '
 'the gate responds to RATE not amount, and the loop never converges over 211 cycles. Stopped at '
 '0 dB with coherence down to 0.697.')
R[51] = ('NL-bs','optimal_diagonal','bounded softening, trim OFF',
 'First optimal-diagonal run on a moving plant, and the baseline for the trim pair. Beats '
 'match_trace decisively on shape -- 1.89 dB better on 38% less drive, coherence 0.953 against '
 '0.697 -- but level error runs +0.31, +1.98 and +8.00 dB at -18, 0 and +4 dB. The +4 dB point is '
 'a level runaway, not a control point.')
R[52] = ('NL-bs','optimal_diagonal','bounded softening, trim ON, pre-fix',
 "The response-error trim's first live test. FAILED its null check at -18 dB, where the true "
 'model gap is 0.012 dB: corrections ended pinned across the +/-3 dB clamp and cost 30% more '
 'drive for the same response. Cause: integrator windup behind a conditional gate evaluated '
 'before the error term. Stopped at -18 dB.')
R[53] = ('--','optimal_diag_fast','bounded softening, trim OFF',
 "NOT RUN. Baseline for the fast law's trim pair on the moving plant. Profiles exist and are "
 "untouched. Note the trim leak constant is tuned to the slow law's refinement rate and is "
 'probably wrong for the fast one.')
R[54] = ('--','optimal_diag_fast','bounded softening, trim ON',
 'NOT RUN. Same status as run 53. Run 55 took its slot as the trim retest on the slow law.')
R[55] = ('NL-bs','optimal_diagonal','bounded softening, trim ON, leak fix',
 'Trim retest after the windup fix. PARTIAL CURE: clamped bins fell from about 1700 to 980, the '
 'trim median moved from -1.51 to +0.00 dB, and 28% of the excess drive was recovered. But at '
 '0 dB the trim is worse than trim-off on every measure. It improved the two channels the solver '
 'had conceded and degraded five of the six it was serving -- it reads a rank-deficiency '
 'trade-off as model error. Also gave the third-harmonic curve, -53.5 to -29.5 dB while in-band '
 'error in the same modes stayed flat.')
R[56] = ('NL-bs','optimal_diagonal','bounded softening, trim OFF, df = 10 Hz',
 'System identification captured at 10 Hz resolution to test control on a wider bandwidth. No '
 'spectral file was saved and nothing was scored. The console log shows control cycles did run, '
 'so this is unsaved rather than unattempted, and worth repeating properly.')
PLANT_NOTE = {'LIN': 'linear', 'NL-c1': 'cubic-1mode', 'NL-call': 'cubic-allmodes',
              'NL-cesc': 'cubic-escapecal', 'NL-bs': 'bounded-soften', '--': '(not run)'}
W = 96


def block(n):
    plant, law, cfg, desc = R[n]
    d = LIN.get('run%02d' % n)
    if d is not None:
        nums = '%7.2f %6.1f %+7.2f' % (d['rms'], d['pout'], d['level'])
    elif plant == '--':
        nums = '      -      -       -'
    else:
        nums = '    (not scored here) '
    head = ' %2d  %-15s %-18s %-30s %s' % (n, PLANT_NOTE[plant], law, cfg[:30], nums)
    body = textwrap.fill(desc, width=W, initial_indent=' ' * 5, subsequent_indent=' ' * 5)
    return head + '\n' + body


HEAD = """\
==================================================================================================
ALL TEST RUNS -- sixdrive12resp
Six-shaker frame, twelve response channels, eight used for control, 100-1000 Hz at 1 Hz.
==================================================================================================
Generated by code/make_run_chart.py. The three numeric columns are read from report_data.json,
not transcribed. Descriptions come from the runsheets and session summaries.

COLUMNS
  rms     in-band per-channel rms error, dB. Lower is better.
  %out    percentage of in-band lines outside the tolerance band.
  level   total level error against specification, dB. A law can score a good rms and still be
          far off level; accuracy and level control are separate problems, and two of the six
          laws fail only the second.

  A dash means the run was never taken. "(not scored here)" means it produced no scored spectrum
  in this dataset -- it either ended before one was saved, or it is a nonlinear run scored on the
  coherence-weighted column in its own notes, because raw rms counts distortion as signal.

PLANTS
  linear           sdynpy_frame6x12_system.npz. 33 flexible modes, 120.0 to 899.5 Hz.
  cubic-1mode      softening cubic on one mode. Too weak to move the FRF at all.
  cubic-allmodes   softening cubic on all 33 modes. Escapes.
  cubic-escapecal  cubic with k3 calibrated per mode against its own escape amplitude. Escapes.
  bounded-soften   g(q) = a + (1-a)exp(-(q/qt)^2), a = 0.35. The 27 modes above 300 Hz soften;
                   the six below are exactly linear. Cannot escape at any level or strength.

 run  plant           law                config                              rms   %out    level
--------------------------------------------------------------------------------------------------"""

TAIL = """
==================================================================================================
WHAT THE SHAPE OF THIS TABLE SAYS
==================================================================================================
56 slots, 40 runs with saved data, 16 never taken. Of the 40, two produced nothing scorable
(13 inadmissible, 14 stopped before saving) and four were nonlinear attempts that escaped or
could not be excited (42, 44, 46, 48). That leaves 34 scored runs on the linear plant and six on
the bounded-softening plant.

The 16 never taken are mostly not a matter of running out of time. Runs 17 and 18 were overtaken
when the dB objective was defaulted off. Runs 36, 37 and 38 were abandoned once run 35 showed
there was nothing left for a trim to correct. Runs 43, 45 and 47 were abandoned as each cubic
article failed in turn. Run 41 was a contingency that run 40 made unnecessary. Only runs 53 and
54 are genuinely outstanding work.

Three laws were carried forward -- optimal_diagonal, its fast variant, and
match_trace_pseudoinverse -- and three were not. congruence has the best raw accuracy in the set
and has never held level on any run. buzz and pseudoinverse are open loop by construction.

Five runs account for most of what was learned about the nonlinear plant: 49 established that it
moves only where it was built to move, 50 that the FRF refresh genuinely tracks it, 51 that
optimal_diagonal tracks shape best and has no level loop at all, and 52 and 55 that the
response-error trim is the wrong instrument for closing it.

==================================================================================================
CORRECTIONS TO EARLIER WRITE-UPS
==================================================================================================
Five claims made during this work were later withdrawn on evidence. They are listed here so that
reading an older document does not re-contaminate the record.

  1. rcond = 1e-3 does NOT cost achievable headroom. The plant is analytically rank 5
     (sigma_6 is about 1e-17), so rcond is rejecting a direction the hardware cannot reach.
  2. Run 13 is inadmissible because the WRONG LAW RAN, not because of a weak system ID. Scored
     like for like, run 13 had the best identification of the set.
  3. The drive-coherence cap's cost is set by WHERE the cap is applied and how much correction
     authority the law retains afterwards, not by loop family.
  4. optimal_diagonal and its fast variant were classified first as feedback and later as open
     loop. Both labels were wrong: they close on the identified plant model and are open on the
     response error.
  5. FOUND WHILE BUILDING THIS CHART. The 2026-09-26 report says of the cubic articles that
     "four runs were attempted on it and all four ended the same way -- arithmetic overflow in
     the time integration". That is wrong for run 42, whose console log is six lines with zero
     overflow warnings; it was stopped after system identification because the single-mode
     article could not be excited enough to move its FRF. Three escaped, one was stopped for
     lack of effect. The 09-23 and 09-24 session summaries have it right.
"""

with open(OUT, 'w') as fh:
    fh.write(HEAD + '\n')
    for n in range(1, 57):
        fh.write(block(n) + '\n')
        if n in (12, 18, 28, 35, 41, 48):
            fh.write('-' * 98 + '\n')
    fh.write(TAIL)
print('wrote', OUT)
