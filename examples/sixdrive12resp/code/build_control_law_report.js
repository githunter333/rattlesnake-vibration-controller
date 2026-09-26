const fs = require('fs');
const d = require('docx');
const {Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
       Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
       TableOfContents, PageOrientation} = d;

const INK = "1A1A1A", MUT = "52514E", ACC = "1F4E79", WARN = "9C3A17";
const LETTER = { size: { width: 12240, height: 15840 }, margin: {top:1080,bottom:1080,left:1080,right:1080} };

const P = (t, o={}) => new Paragraph({
  spacing:{after:o.after??120, before:o.before??0, line:o.line??276},
  alignment:o.align, heading:o.heading, pageBreakBefore:o.pb,
  border:o.rule?{bottom:{style:BorderStyle.SINGLE,size:6,color:"D9D8D4",space:6}}:undefined,
  children:[new TextRun({text:t, bold:o.b, italics:o.i, size:o.size??20,
                         color:o.color??INK, font:o.font??"Calibri"})]});

const RUNS = (parts, o={}) => new Paragraph({
  spacing:{after:o.after??120, before:o.before??0, line:276}, alignment:o.align,
  children: parts.map(p => new TextRun({text:p.t, bold:p.b, italics:p.i,
    size:p.size??20, color:p.color??INK, font:p.font??(p.mono?"Consolas":"Calibri")}))});

const H1 = (t,pb=false) => new Paragraph({heading:HeadingLevel.HEADING_1, pageBreakBefore:pb,
  spacing:{before:pb?0:320, after:140},
  children:[new TextRun({text:t, bold:true, size:30, color:ACC, font:"Calibri"})]});
const H2 = t => new Paragraph({heading:HeadingLevel.HEADING_2, spacing:{before:260,after:110},
  children:[new TextRun({text:t, bold:true, size:24, color:INK, font:"Calibri"})]});
const H3 = t => new Paragraph({heading:HeadingLevel.HEADING_3, spacing:{before:200,after:90},
  children:[new TextRun({text:t, bold:true, size:21, color:MUT, font:"Calibri"})]});

function fig(file, caption, widthPt) {
  const w = widthPt ?? 470;
  const img = fs.readFileSync(file);
  // preserve aspect: read PNG header for dimensions
  const iw = img.readUInt32BE(16), ih = img.readUInt32BE(20);
  const h = Math.round(w * ih / iw);
  return [
    new Paragraph({spacing:{before:160,after:40}, alignment:AlignmentType.CENTER,
      children:[new ImageRun({type:"png", data:img, transformation:{width:w, height:h}})]}),
    new Paragraph({spacing:{after:200}, alignment:AlignmentType.CENTER,
      children:[new TextRun({text:caption, size:17, italics:true, color:MUT, font:"Calibri"})]})
  ];
}

function table(headers, rows, widths, opts={}) {
  const total = widths.reduce((a,b)=>a+b,0);
  const cell = (txt, o={}) => new TableCell({
    width:{size:o.w, type:WidthType.DXA},
    shading:o.head?{type:ShadingType.CLEAR, fill:"EEF3F8", color:"auto"}
                  :(o.alt?{type:ShadingType.CLEAR, fill:"F8F8F7", color:"auto"}:undefined),
    margins:{top:60,bottom:60,left:90,right:90},
    children:[new Paragraph({spacing:{after:0,line:240}, alignment:o.align,
      children:[new TextRun({text:String(txt), bold:o.head||o.b, size:o.size??17,
        color:o.color??(o.head?INK:INK), font:o.mono?"Consolas":"Calibri"})]})]});
  const trs = [new TableRow({tableHeader:true, children:headers.map((h,i)=>
    cell(h,{head:true,w:widths[i],align:i?AlignmentType.CENTER:AlignmentType.LEFT}))})];
  rows.forEach((r,ri)=>trs.push(new TableRow({children:r.map((c,i)=>{
    const spec = (typeof c==='object' && c!==null) ? c : {t:c};
    return cell(spec.t,{w:widths[i], alt:ri%2===1, mono:opts.mono&&i>0,
      align:i?AlignmentType.CENTER:AlignmentType.LEFT, b:spec.b, color:spec.color});
  })})));
  return new Table({columnWidths:widths, width:{size:total,type:WidthType.DXA}, rows:trs,
    borders:{top:{style:BorderStyle.SINGLE,size:4,color:"C9C8C4"},
             bottom:{style:BorderStyle.SINGLE,size:4,color:"C9C8C4"},
             left:{style:BorderStyle.NONE},right:{style:BorderStyle.NONE},
             insideHorizontal:{style:BorderStyle.SINGLE,size:2,color:"E4E3DF"},
             insideVertical:{style:BorderStyle.NONE}}});
}

const bullets = (items) => items.map(t => new Paragraph({
  bullet:{level:0}, spacing:{after:70,line:276},
  children:[new TextRun({text:t, size:20, color:INK, font:"Calibri"})]}));

const K = []; // document children

// ============================== TITLE ==============================
K.push(new Paragraph({spacing:{before:1400,after:60}, alignment:AlignmentType.LEFT,
  children:[new TextRun({text:"MIMO Random Vibration Control Laws", bold:true, size:48, color:ACC, font:"Calibri"})]}));
K.push(new Paragraph({spacing:{after:340}, children:[new TextRun({
  text:"Modifications, measured effects, and behaviour on a level-dependent plant", size:26, color:MUT, font:"Calibri"})]}));
K.push(P("Rattlesnake vibration controller — six-drive / twelve-response frame", {size:21, color:MUT}));
K.push(P("Work of 5 – 26 September 2026 · 62 commits · 40 instrumented runs", {size:21, color:MUT, after:340}));
K.push(P("Scope", {b:true, size:22, after:80}));
K.push(P("This report covers three weeks of control-law development on an 8-control-channel / 6-drive " +
  "frame. It is organised in two parts. Part I is the law-by-law record: what each law does, what was " +
  "changed in it, and what each change measurably bought or cost on the linear plant. Part II covers the " +
  "nonlinear plant built to test those laws against a structure whose FRF moves with level.", {after:120}));
K.push(P("Every number quoted is measured from a saved run unless explicitly labelled as a prediction or a " +
  "model calculation. Three claims made earlier in this period were subsequently withdrawn on evidence; " +
  "they are flagged where they arise so the record is not re-contaminated.", {after:200}));
K.push(new Paragraph({children:[new PageBreak()]}));

// ============================== CONTENTS ==============================
K.push(H1("Contents"));
[["1","The plant, and why its rank matters"],
  ["2","All six laws on the linear plant \u2014 the short version"],
  ["3","What was changed in each law, and what it did"],
  ["4","Tracking a moving plant \u2014 what the nonlinear work showed"],
  ["5","A plant whose FRF moves with level"],
  ["6","The laws on the moving plant"],
  ["7","What is open"]].forEach(([n,t]) => K.push(RUNS(
    [{t:n+"\u2002\u2002",b:true,color:ACC},{t:t}], {after:90})));
K.push(new Paragraph({children:[new PageBreak()]}));

// ============================== 1. THE PLANT ==============================
K.push(H1("1  The plant, and why its rank matters"));
K.push(P("The test article is a six-shaker frame with twelve response channels, of which eight are used " +
  "for control, over a 100–1000 Hz band at 1 Hz resolution — 901 in-band lines. Every result in this " +
  "report is bounded by one property of it."));
K.push(RUNS([{t:"The 8×6 plant has rank 5, not 6. "},{t:"Computed analytically from the model's mass, " +
  "stiffness and damping matrices — validated against the measured FRF, which it matches to a median " +
  "0.21 dB in magnitude — the sixth singular value is ~1e-17. The six shakers span a five-dimensional " +
  "response subspace at these eight control locations. In the measured FRF that direction returns at " +
  "3.9e-05 to 4.1e-04 of the first singular value, and the numeric rank over the control band is 5 in " +
  "all twelve runs of the reference set. It is identification noise."}]));
K.push(...fig("fig3_singular_values.png",
  "Figure 1 — Singular values of the control FRF, median over the in-band lines, normalised to σ₁. " +
  "The analytic model and the measured identification agree to a few percent on σ₁…σ₅ and diverge " +
  "entirely on σ₆, which exists only in the measurement."));
K.push(RUNS([{t:"Withdrawn claim. ",b:true,color:WARN},{t:"An earlier version of this work attributed " +
  "roughly 1.6 dB of the achievable-accuracy floor to the rcond = 1e-3 truncation and made relaxing " +
  "rcond the first recommended action. That was wrong. Truncating the measured FRF to rank 5 and " +
  "recomputing with no rcond restriction gives 3.56–3.90 dB against an operational floor within 0.21 dB " +
  "at every run. rcond = 1e-3 is not costing headroom; it is rejecting a direction the hardware cannot " +
  "reach, and lowering it would invert noise."}], {after:160}));
K.push(P("Two consequences run through everything below. Eight channels cannot be independently " +
  "controlled by five usable directions, so every law must decide which channels to sacrifice — and the " +
  "laws differ mainly in how they make that decision. And every accuracy figure in this report is a " +
  "rank-5 floor, not an absolute one."));

// ============================== 2. ALL LAWS, LINEAR ==============================
K.push(H1("2  All six laws on the linear plant — the short version", true));
K.push(P("Six laws were exercised on the linear plant across 34 instrumented runs. The summary below is " +
  "the best each achieved, and the qualification that matters is that accuracy and level control are " +
  "separate problems: two laws never held total level at all."));
K.push(table(
  ["Law","Loop","Best rms","% out","Level held?","Verdict"],
  [
   ["optimal_diagonal","model","5.62 dB","11.2","yes (+0.08)",{t:"carry forward",b:true}],
   ["optimal_diagonal_fast","model","5.64 dB","10.9","yes (+0.21)",{t:"carry forward",b:true}],
   ["match_trace_pseudoinverse","model + error","7.21 dB","21.1","yes (+0.05)",{t:"carry forward",b:true}],
   ["diagonal_congruence","error","5.23 dB","18.7",{t:"no (best 2.78)",color:WARN},"promising, level unsolved"],
   ["buzz","none","5.91 dB","13.6","yes (−0.30)","open loop, no tracking"],
   ["pseudoinverse","none","7.01 dB","17.8",{t:"no (best 1.28)",color:WARN},"open loop, no tracking"],
  ],
  [2600,1150,1000,700,1400,2050]));
K.push(P("“Loop” is what the law closes on. model = re-solves against the live FRF every cycle; " +
  "error = reads the measured response; none = neither. Best rms is the lowest in-band per-channel rms " +
  "error achieved while holding total level within ±0.5 dB, except where marked, where no run held level " +
  "and the unconstrained best is shown instead.", {size:18, color:MUT, after:160}));
K.push(...fig("fig2_rms_vs_level.png",
  "Figure 2 — Every run on the linear plant. Accuracy on the vertical axis, level control on the " +
  "horizontal. The two are largely independent: the best accuracy in the set sits outside the level " +
  "acceptance band, and the laws that hold level cluster on the left regardless of accuracy."));
K.push(H2("2.1  What this says about the three laws worth carrying"));
K.push(...bullets([
  "optimal_diagonal and its fast variant give the best accuracy that is simultaneously achievable with " +
  "correct level — 5.62 and 5.64 dB — and do it on roughly a third of the drive power the other laws need. " +
  "They concede two of the eight channels to buy the other six.",
  "match_trace_pseudoinverse is 1.6 dB worse but is the only law of the three that reads the measured " +
  "response, and its level control is the tightest in the set. Its per-line correction is a single real " +
  "scalar, which is both its safety property and its accuracy ceiling.",
  "diagonal_congruence reaches the best raw accuracy of any law, 5.23 dB, correcting every channel rather " +
  "than the total. It has never held level — its best is 2.78 dB — and that is the open problem on it, " +
  "not its accuracy.",
]));
K.push(RUNS([{t:"Withdrawn claim. ",b:true,color:WARN},{t:"Two laws were at one point classified as " +
  "feedback and later as open loop; both classifications were wrong. optimal_diagonal and its fast " +
  "variant are closed on the identified plant model — they re-solve against the live FRF every cycle — " +
  "and open on the response error, since the objective is the predicted response diag(H X Hᴴ) and " +
  "last_response_cpsd is accepted and never read. The three-way classification in the table above is " +
  "the corrected one."}]));

// ============================== 3. MODIFICATIONS ==============================
K.push(H1("3  What was changed in each law, and what it did", true));
K.push(P("This is the substance of the three weeks. The changes fall into four groups: the startup level " +
  "cap, the drive-coherence cap, making the FRF update real, and per-law guards. A fifth entry records the " +
  "decision to default every behaviour change back off."));

K.push(H2("3.1  The startup level cap — two bugs, present in every law that had one"));
K.push(P("The guard clamps the first control command to a ceiling relative to specification, so the rig is " +
  "not hit with an unbounded first shot. It was present in six laws and wrong in both of the ways it could be."));
K.push(RUNS([{t:"Wrong domain. ",b:true},{t:"The clamp compared tr(specification) — a response quantity — " +
  "against tr(output), which is the drive CPSD in V². That ratio is not dimensionless: it carries units of " +
  "(response/volt)², so its value tracked the plant gain and the response channels' units rather than the " +
  "overshoot it was meant to bound. On run 01's identification the raw solve sat −2.04 dB re spec; the " +
  "clamp as written took it to a median −18.88 dB, where the clamp as documented takes it to exactly " +
  "−9.00 dB. Roughly 10 dB too aggressive, and unit-dependent. It now clamps on the predicted response, " +
  "tr(H X Hᴴ)."}]));
K.push(RUNS([{t:"Wrong order. ",b:true},{t:"The drive-coherence cap ran after the clamp and lifted the " +
  "level straight back through the ceiling. The cap shrinks off-diagonal drive terms while preserving the " +
  "diagonal, which destroys the inter-drive cancellation a pseudoinverse solve relies on, so the same " +
  "drive produces far more response. Both were fixed in all six laws, and the four laws that had no " +
  "ceiling at all were given one."}]));
K.push(P("This is the one behaviour change left enabled by default, at −9 dB, on every law.", {i:true, color:MUT}));

K.push(H2("3.2  The drive-coherence cap — free as a constraint, costly as a correction"));
K.push(P("The 0.95 cap on drive coherence exists to keep the commanded drives from becoming near-identical, " +
  "which is both physically unrealisable and numerically fragile. Its cost was measured law by law with " +
  "matched cap-on / cap-off pairs."));
K.push(...fig("fig1_cap_cost.png",
  "Figure 3 — Accuracy cost of the 0.95 drive-coherence cap. The determining factor is not the law family " +
  "but where the cap is applied and how much authority the law retains afterwards."));
K.push(P("optimal_diagonal solves subject to the cap as an SDP constraint, so the returned solution is " +
  "already the best one that honours it and the cap costs essentially nothing. Every other law " +
  "post-processes the cap onto an already-computed solve, and what it then costs depends on how much " +
  "correction authority is left: congruence re-converges per drive and loses 0.08 dB; match_trace has only " +
  "a scalar per line and loses 1.01 dB; buzz and pseudoinverse have no correction mechanism at all and " +
  "lose 1.99 and 3.47 dB."));
K.push(RUNS([{t:"Withdrawn claim. ",b:true,color:WARN},{t:"An earlier version explained this spread by " +
  "loop family. That explanation depended on the mistaken loop classification and never fitted the " +
  "measurements anyway. Where the cap is applied is what fits."}]));

K.push(H2("3.3  Making “Update Transfer Function During Control” real"));
K.push(P("An audit of what an updated FRF actually does to each law found three where the setting did not " +
  "work as anyone would assume, and one place where it was actively unsafe. The call contract had been " +
  "misunderstood: Rattlesnake passes a transfer_function to control() every cycle whether the setting is " +
  "on or off — the setting only chooses which array, live or frozen — and system_id_update() is never " +
  "called during control at all. A law that reacts to a new FRF only inside system_id_update is therefore " +
  "a silent no-op under the setting."));
K.push(RUNS([{t:"match_trace ignored the updated FRF entirely. ",b:true},{t:"Its steady state is a " +
  "per-line real scalar applied to the previous command, so the FRF entered once at startup and never " +
  "again: it tracked level against a re-estimated plant while holding a drive shape synthesised from the " +
  "plant as it looked before the test started. It now re-synthesises the shape from the current FRF and " +
  "carries the level forward explicitly."}]));
K.push(...fig("fig4b_frf_update.png",
  "Figure 4 — optimal_diagonal_fast with the FRF update working. Three repeats land within ±7% of " +
  "17.2 V² and on the offline prediction of 16.20 V². With the update off, the same commanded solution " +
  "gave 153, 485 and 245 V² across repeats."));
K.push(P("That reproducibility is the point. An unreproducible drive is not a control law, it is a " +
  "coincidence; making the update real is what turned the fast law from an interesting solver into " +
  "something that can be tested."));

K.push(H2("3.4  Per-law guards and refinements"));
K.push(H3("match_trace_pseudoinverse"));
K.push(...bullets([
  "Gated FRF refresh. The refresh is gated per line on change since that line's own last solve, not since " +
  "the previous cycle — which can never see slow monotonic drift. Ungated refresh re-solves on estimator " +
  "noise, because the published FRF is already exponentially averaged. Best result on the linear plant " +
  "improved from 8.17 to 7.37 dB rms with the gate at 0.05.",
  "Gap-based rank selection. Rather than an absolute rcond, the truncation point is located by the largest " +
  "gap in the singular-value spectrum — 8.9× between σ₅ and σ₆ against 2.1–3.8× elsewhere. Cross-estimate " +
  "agreement rose from 24.5% to 88.7%.",
  "The drive-shape refresh defaults OFF, so the reference runs reproduce.",
]));
K.push(H3("optimal_diagonal and optimal_diagonal_fast"));
K.push(...bullets([
  "drive_rcond expressed as a multiple of the plant's own singular-value spread, measured from the system " +
  "ID the law already holds, rather than an absolute ratio that must be re-tuned per article. The reference " +
  "is σ₅/σ₁ = 0.0152, located by the spectral gap; default 2.0. Two errors were made and fixed building " +
  "it — measuring the spread over all lines rather than in-band only, and scaling off σ_min rather than σ₅.",
  "A dB-domain objective as an alternative to the linear one. It gives better pooled rms but spreads error " +
  "across every channel instead of sacrificing the two the plant cannot reach, which is not what this rig " +
  "wants. Opt-in; the default is unchanged.",
  "A response-error trim, to close the loop the law leaves open. See §6.3 — it does not earn its place.",
]));
K.push(H3("diagonal_congruence"));
K.push(...bullets([
  "The anti-windup ceiling now judges level from the measured response extrapolated over one bounded step, " +
  "not from the response predicted through H. The principal limit was otherwise only as trustworthy as the " +
  "model — exactly what cannot be relied on when the FRF is being re-estimated live, which is the regime " +
  "the earlier laws failed in.",
  "The projected target is read lazily on the first control() call rather than in __init__, which runs " +
  "before system identification. Reading it in __init__ demanded a file derived from an FRF that does not " +
  "yet exist, so the operator could only ever supply a stale projection.",
  "Its startup clamp was already in the response domain, but ran before the floor-and-silence and " +
  "drive-coherence steps, both of which lift the level back through the ceiling. Reordered.",
]));

K.push(H2("3.5  Every behaviour change defaulted back off"));
K.push(P("Several of these changes silently altered the laws that produced the twelve-run reference set. " +
  "Each was made opt-in so that the defaults reproduce those runs exactly, with one exception: the startup " +
  "level cap at −9 dB stays on, because it is wanted. This is the discipline that makes the comparison " +
  "table in §2 meaningful — without it, every improvement would have quietly invalidated the baseline it " +
  "was measured against."));

// ============================== 4. TRACKING SUMMARY ==============================
K.push(H1("4  Tracking a moving plant \u2014 what the nonlinear work showed", true));
K.push(P("The second half of the period built a plant whose FRF moves with drive level and ran the " +
  "surviving laws against it. \u00a75 describes how that plant was built and \u00a76 gives the run-by-run " +
  "detail; this section is the answer to the question the whole exercise was for \u2014 how well does each " +
  "law update as the plant changes underneath it."));

K.push(H2("4.1  A law updates in two places, and they are not the same thing"));
K.push(P("Every law in the set re-solves the drive from a plant estimate on every control cycle. That is " +
  "not plant tracking. Plant tracking requires the estimate itself to be replaced from control-time data, " +
  "which is a separate switch \u2014 Update Transfer Function During Control \u2014 and until this period it " +
  "was a silent no-op in two of the three laws (\u00a73.3). The distinction matters because the two " +
  "updates fail differently: a stale estimate degrades gracefully, while a live estimate fed by distorted " +
  "control-time data can degrade the very model it is meant to improve."));
K.push(P("The nonlinear plant separates the two cleanly, because 27 of its 33 modes soften and six do " +
  "not. Refreshing the estimate below 300 Hz, where nothing moves, buys 0.62 dB \u2014 that is the " +
  "estimator refining a fixed plant. Above 300 Hz the same refresh buys 1.5\u20132.6 dB, and that " +
  "increment is genuine plant tracking. The two jobs separate in frequency only because the control group " +
  "was built to make them."));

K.push(H2("4.2  Law by law, on a plant that moves"));
K.push(table(
  ["Law","Estimate update","Runs","How it behaved as the plant moved"],
  [
   ["match_trace_pseudoinverse",{t:"live, gated",b:true},"49, 50",
    "Tracks. Better at every level with the refresh on; accuracy flat with level. Pays in coherence."],
   ["optimal_diagonal",{t:"live",b:true},"51, 52, 55",
    "Tracks shape best of any law \u2014 and loses level control completely by +4 dB."],
   ["optimal_diagonal_fast",{t:"live",b:true},{t:"none",color:WARN},
    {t:"Untested on a moving plant. Profiles ready.",color:WARN}],
   ["diagonal_congruence",{t:"live",b:true},{t:"none",color:WARN},
    {t:"Untested. Never held level on the linear plant, so a moving one is premature.",color:WARN}],
   ["buzz / pseudoinverse",{t:"none",color:WARN},{t:"none",color:WARN},
    "Open loop by construction. Cannot track and were not asked to."],
  ],
  [2400,1500,1100,5080]));

K.push(H3("match_trace_pseudoinverse \u2014 tracking works, and it is not free"));
K.push(P("The clean pair is runs 49 and 50: one parameter different, four levels each. With the refresh " +
  "on, coherent-column accuracy is 7.40, 7.41, 7.38 and 7.47 dB from \u221218 dB to 0 dB \u2014 a 0.09 dB " +
  "spread while the plant softens 0.745% and the drive voltage spans 12.7\u00d7. With the refresh off the " +
  "same law runs 7.91, 7.95, 7.99 and 8.10 dB. The refresh is worth 0.52, 0.55, 0.62 and 0.63 dB, and the " +
  "margin widens with level, which is the signature of tracking rather than of a fixed offset. Lines out " +
  "of band improve from about 26% to 20\u201323% at every level."));
K.push(P("The cost appears where the estimate is fed by control-time data on a distorting plant. At 0 dB " +
  "coherence falls from 0.938 to 0.697, out-of-band response rises from \u221230.8 to \u221224.5 dB, and " +
  "the drive rises 2.8 dB. The loop also never settles: 211 cycles with the per-cycle FRF change held at " +
  "0.020\u20130.031 against 0.007 with the refresh off. Tracking bought accuracy and spent model quality " +
  "to get it."));

K.push(H3("optimal_diagonal \u2014 the best tracker and the worst level control"));
K.push(P("At 0 dB it reaches 5.58 dB against match_trace\u2019s 7.47 on the same plant, using 38% less " +
  "drive and holding coherence at 0.953 where match_trace falls to 0.697. Asking less of the plant keeps " +
  "the plant closer to the model it is being identified against, so the tracking chain reinforces itself " +
  "instead of eroding. On shape, this is the law to carry."));
K.push(RUNS([{t:"And it does not track level at all. ",b:true},{t:"Level error runs +0.31 dB at " +
  "\u221218, +1.98 at 0 and +8.00 at +4. At the +4 dB point the drive rose 0.9 dB for a 4 dB increase in " +
  "demand while the response overshot by eight, because the law closes on its own predicted response and " +
  "never reads the measured one. On a linear plant that costs nothing. On a softening plant it is a " +
  "runaway, and it is the outstanding defect in the whole family."}]));

K.push(H3("What was tried to close that loop, and why it failed"));
K.push(P("The response-error trim exists to close exactly that loop and had never run live. It ran twice " +
  "on the moving plant and was worse on every measure at the one level where it had real work to do " +
  "(\u00a76.3). The cause is structural rather than a tuning problem: on a rank-5 plant the solver " +
  "deliberately concedes channels it cannot reach, and a trim that reads residual response error as model " +
  "error buys those channels back at full price. Level tracking on this family needs an outer loop on " +
  "total power, not a per-channel correction."));

K.push(H2("4.3  The limit that will matter on hardware"));
K.push(P("The refresh gate responds to the rate of plant change, not the amount. A level change ramped " +
  "over 2 s and smoothed by 0.04 exponential averaging arrives as roughly 25 increments fifty times under " +
  "the 0.05 threshold, and two of the four level changes in run 50 produced no refresh activity at all " +
  "where 321 and 447 lines had been predicted. The tracking demonstrated above therefore happened on the " +
  "changes that were fast enough to be seen. A real structure heating or loosening over a test would " +
  "drift arbitrarily far under this gate without ever tripping it. An accumulated-change criterion " +
  "alongside the per-cycle one is the single change most needed before any of this meets hardware."));
K.push(P("One further caution carries from \u00a75.3: on a moving plant the scoring metric is itself " +
  "suspect. Raw rms counts distortion as signal and flattered one run by 1.55 dB; rms about zero partly " +
  "cancels on a uniformly hot response, so the +4 dB runaway scores better than the converged 0 dB point. " +
  "Every accuracy number above is on the coherence-weighted column, read alongside level error."));

// ============================== 5. NONLINEAR PLANT ==============================
K.push(H1("5  A plant whose FRF moves with level", true));
K.push(P("Every result above is on a linear plant. Real structures are not linear, and the laws that track " +
  "an FRF should be tested against one that actually moves. That required building a nonlinear plant, " +
  "which took two attempts."));

K.push(H2("5.1  Why the first attempt failed four times"));
K.push(P("The first nonlinear article used a softening cubic in each modal coordinate. Four runs were " +
  "attempted on it and all four ended the same way — arithmetic overflow in the time integration. That is " +
  "escape, not a jump: a softening cubic gives a finite potential well, and beyond q_esc = ωₙ/√|k₃| the " +
  "restoring force reverses and the solution runs away."));
K.push(RUNS([{t:"The trap is worth stating, because it cost four runs. ",b:true},{t:"You cannot buy " +
  "frequency shift by turning up k₃. The shift rises linearly with |k₃| while q_esc falls as 1/√|k₃|, so " +
  "the shift available at escape is fixed at 3/8 whatever k₃ is. With 4σ random peaks that caps the usable " +
  "shift near 0.5–1%, and the final calibration achieved 0.375% — at the noise."}]));
K.push(...fig("fig4_kernel.png",
  "Figure 5 — The two kernels. The cubic's restoring force reverses sign at the escape amplitude, giving a " +
  "finite well. The bounded kernel saturates onto a residual linear stiffness αωₙ²q and never reverses, so " +
  "the potential is unbounded and there is nothing to escape from.", 440));
K.push(P("The replacement, a bounded softening kernel g(q) = α + (1−α)·exp(−(q/q_t)²), removes the failure " +
  "mode structurally rather than by margin. Verified by direct integration at 400× the design nonlinearity " +
  "and 4× the maximum test level: finite, with the frequency shift saturating at 40.84% — exactly the " +
  "1 − √α asymptote. The cubic escaped at 25× design."));

K.push(H2("5.2  The design decision that made the results readable"));
K.push(P("All 33 flexible modes lie between 120 and 899.5 Hz. Rather than soften all of them, only the 27 " +
  "above 300 Hz were made nonlinear; the six below stay exactly linear — their nonlinear term is " +
  "identically zero, measured at 2e-13% shift, not merely small."));
K.push(...fig("fig5_control_group.png",
  "Figure 6 — Run 49. The six linear modes and the 27 softened modes, measured as per-line FRF shape " +
  "change against the −18 dB state. The linear band is flat across a 12.7× span of drive voltage while the " +
  "softened band rises 18×."));
K.push(P("Those six modes cost nothing — they were already there — and they convert “the FRF changed” into " +
  "“the FRF changed in exactly the 27 modes we made nonlinear and in none of the six we did not”, inside a " +
  "single run. They paid again when the FRF refresh was tested: below 300 Hz, where nothing moves, the " +
  "refresh still buys 0.62 dB, which is estimate refinement; above 300 Hz it buys 1.5–2.6 dB, and that " +
  "increment is plant tracking. The two functions separate in frequency only because of the control group."));

K.push(H2("5.3  The measurement lesson"));
K.push(P("Control quality appeared to improve monotonically with level — rms 7.89 dB at −18 dB falling to " +
  "4.96 dB at +8 dB — while the plant moved further from the frozen FRF the drive was solved against. " +
  "That is backwards, and it is an artifact. Weighting the response auto-spectra by the measured coherence " +
  "and scoring only the part correlated with the drive, the improvement vanishes: control is flat to " +
  "slightly worse, 7.9 to 8.3 dB."));
K.push(P("The extra “signal” was distortion. The scoring metric cannot distinguish driven response from " +
  "distortion, and neither can the law — match_trace is matching the trace of a response that is partly " +
  "not its own. Every accuracy number in this report is therefore taken on the coherence-weighted column. " +
  "Raw rms flattered one run by 1.55 dB."));
K.push(...fig("fig6_third_harmonic.png",
  "Figure 7 — Run 55. Out-of-band third-harmonic content against in-band error, in the same modes. " +
  "1250–1450 Hz is three times the most-softened mode cluster; the plant has no modes above 899.5 Hz, so " +
  "nothing there is resonant."));
K.push(P("This is the most portable finding of the three weeks. On this rig the nonlinearity appears in " +
  "the out-of-band harmonic long before it appears in the in-band control error — 24 dB of range against " +
  "0.30 dB. The in-band error is set by the rank-5 geometry, not by the plant moving. The harmonic needs " +
  "no reference run and is free to measure from any saved spectrum."));

// ============================== 5. LAWS ON THE NONLINEAR PLANT ==============================
K.push(H1("6  The laws on the moving plant", true));
K.push(H2("6.1  match_trace with and without the FRF refresh"));
K.push(P("A clean pair — one parameter different — at four levels. On the coherent column the " +
  "refresh is better at every level, by 0.52, 0.55, 0.62 and 0.63 dB. With it on, accuracy is 7.40, 7.41, " +
  "7.38 and 7.47 dB across the sweep; with it off, 7.91, 7.95, 7.99 and 8.10 dB. Both configurations are " +
  "nearly flat with level, but the gap widens as the plant moves, which is what tracking should look " +
  "like. The refresh-on result at 0 dB, on a plant softened by 0.745%, is better than the refresh-off " +
  "result at −18 dB on the same plant when it was effectively linear — the two differ in " +
  "configuration as well as level, so that comparison bounds what the refresh is worth rather than " +
  "claiming the softened plant is intrinsically easier."));
K.push(P("Two costs. Coherence falls from 0.938 to 0.697 as control-time estimates carrying distortion are " +
  "fed back into the solve, and the loop never converges — 211 cycles with the per-cycle FRF change held at " +
  "0.020–0.031 against 0.007 with the refresh off, and the refresh count parked at a median of 300 with no " +
  "decay. Three coupled time constants with nothing damping between them."));
K.push(RUNS([{t:"An unanticipated property. ",b:true},{t:"The gate responds to the rate of plant change, " +
  "not the amount. A level change ramped over 2 s and smoothed by 0.04 exponential averaging arrives as " +
  "roughly 25 increments fifty times under the 0.05 threshold, so two of the four level changes produced " +
  "no refresh activity at all where 321 and 447 lines had been predicted. A slowly drifting real structure " +
  "would slip under this gate no matter how far it eventually drifted. This wants an accumulated-change " +
  "criterion alongside the per-cycle one before the law sees hardware."}]));

K.push(H2("6.2  optimal_diagonal on the moving plant"));
K.push(P("At 0 dB it beats the best match_trace result by 1.89 dB on the coherent column — 5.58 against " +
  "7.47 — using 38% less drive, and holds coherence at 0.953 where match_trace falls to 0.697. Less drive " +
  "means less distortion, which means a cleaner FRF, which means better control; the chain follows from " +
  "asking less of the plant."));
K.push(P("Its weakness is level. The level error runs +0.31, +1.98 and +8.00 dB at −18, 0 and +4 dB. At " +
  "+4 dB the response is eight dB above specification and the law is blind to it: the drive rose only " +
  "0.9 dB for a 4 dB increase in demand while the response overshot by 8. That is the open-on-response-error " +
  "property, not as a subtlety but as an eight dB miss. That point is a level runaway, not a control point."));
K.push(RUNS([{t:"A scoring trap worth carrying. ",b:true},{t:"The coherent rms at that +4 dB point reads " +
  "5.53 dB — better than the same run's 0 dB result — while the test is eight dB over specification, " +
  "because rms-about-zero partly cancels on a uniformly hot response. rms alone cannot see a level " +
  "runaway. Read level error alongside it, always."}]));

K.push(H2("6.3  The response-error trim: a negative result"));
K.push(P("The trim was built to close exactly the loop §6.2 shows open, and had never run live. It ran " +
  "twice. At 0 dB — the one level where the model gap gives it real work — it is worse on every measure " +
  "than the same law with the trim off: 0.10 dB worse on the coherent column, 1.7 points more lines out of " +
  "band, 0.73 dB further off level, 25% more drive, and coherence down from 0.953 to 0.896."));
K.push(P("The per-channel breakdown says why. The trim improved the two channels the solver had " +
  "conceded — 10.34 to 9.20 dB and 7.71 to 7.42 dB — and degraded five of the six it was serving, " +
  "the sixth improving by 0.13 dB. It pulls drive toward channels the SDP deliberately gave up on, " +
  "because it reads a rank-deficiency trade-off as model error. The solver concedes those channels because " +
  "serving them costs more elsewhere; the trim cannot see that trade and buys them back at full price."));
K.push(P("That is a design mismatch rather than a tuning problem, and it is the reason the remaining " +
  "windup repair is now low priority — a perfect trim would still buy back conceded channels.", {i:true}));
K.push(H3("The windup, and four fixes"));
K.push(P("The trim also failed its own null check, ending with corrections pinned across the ±3 dB clamp " +
  "at a level whose true model gap is 0.012 dB, and costing 30% more drive for the same response. The " +
  "cause is integrator windup behind a conditional gate: the solver-agreement gate is evaluated before the " +
  "error term, and gated bins kept their correction permanently, so the gate blocked harmful accumulation " +
  "and corrective unwinding equally."));
K.push(table(
  ["Attempt","Change","Outcome"],
  [
   ["1","Leak every idle bin",{t:"Steady state sat at the deadband edge — a limit cycle",color:WARN}],
   ["2","Leak on any single gate failure",{t:"Bled away corrections merely waiting to be re-solved",color:WARN}],
   ["3","Leak after 50 consecutive gate failures",{t:"Committed. Partial cure — 28% of the excess drive recovered",b:true}],
   ["4","Pay down the counter instead of resetting",{t:"Swamped a real 3 dB model error. Reverted",color:WARN}],
  ],
  [900,3200,4700]));
K.push(P("All four attempts assumed that gate-failure frequency identifies a spurious correction. It " +
  "cannot: with 20 bins re-solved per cycle out of 901, a bin waiting its turn fails the gate about 98% of " +
  "the time and so does a bin the solver genuinely cannot reach. The discriminator is whether a bin passes " +
  "on the cycles it is actually re-solved, which requires the refinement scheduler to tell the trim which " +
  "bins it touched. Three of the four were caught by the regression suite rather than by reasoning."));

// ============================== 7. OPEN ==============================
K.push(H1("7  What is open", true));
K.push(H2("7.1  Ready to run"));
K.push(...bullets([
  "optimal_diagonal_fast on the nonlinear plant. Profiles exist and are untouched. Note that the trim's " +
  "leak constant is tuned against the slow law's 20-bins-per-cycle refinement rate and is probably wrong " +
  "for the fast variant.",
  "A Δf = 10 Hz system identification was captured as run 56 to test control on a wider bandwidth, " +
  "but the environment was never run against it. The sysid file is saved.",
  "octave_band_switching_control has never been run. Its parameter positions collide with the base class's " +
  "— it reads fields 5–9 as its own where the base now reads drive_rcond and the startup cap — so it needs " +
  "a five-line fix before it can work at all. This is the vehicle for wider-bandwidth control, keeping fine " +
  "acquisition and aggregating only the control decisions into bands.",
]));
K.push(H2("7.2  Known defects"));
K.push(...bullets([
  "The level runaway in the optimal-diagonal family is the real outstanding defect, and the trim is not " +
  "the fix. A level-only outer loop reading total power would be.",
  "The refresh gate is rate-sensitive and will miss slow drift; it wants an accumulated-change criterion.",
  "diagonal_congruence has never held level on any run.",
]));
K.push(H2("7.3  Unrepeated"));
K.push(P("Both the match_trace pair and the optimal_diagonal pair on the nonlinear plant are one run each. " +
  "Run 49's +4 dB down-sweep is unresolved as a hysteresis question — the two +4 dB points differ, but the " +
  "level loop landed 1.31 dB apart, which accounts for most of it."));

K.push(H2("7.4  Portability"));
K.push(P("The three laws will run on stock Rattlesnake: the control-law interface is byte-identical to " +
  "upstream, and everything the laws consume — the live control FRF, the last response and drive CPSDs, " +
  "the update-during-control switch, and the test-level normalisation — is upstream-native. They import " +
  "only numpy. Three things do not travel: the nonlinear virtual hardware is a local component and the " +
  "profiles select it by a hardware index upstream does not define; the exponential control-phase " +
  "averaging is a local profile field; and the level-change hook the trim and the octave law use is never " +
  "called upstream. Portable as code, not as results."));

// ============================== BUILD ==============================
const doc = new Document({
  creator:"Rattlesnake control-law development",
  title:"MIMO Random Vibration Control Laws",
  description:"Modifications, measured effects, and behaviour on a level-dependent plant",
  styles:{default:{document:{run:{font:"Calibri", size:20, color:INK}}}},
  sections:[{properties:{page:LETTER}, children:K}]});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("control_law_report_2026-09-26.docx", b);
  console.log("written", b.length, "bytes");
});
