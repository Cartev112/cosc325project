# COSC 325 Project Proposals

This document presents four EEG-focused machine-learning project candidates. Each proposal combines a defensible scientific question, an open dataset, a classical ML evaluation plan, and an interactive system demonstration.

## Project Pitches

### 1. Attention Time Machine

**A neuroadaptive reader that finds the sentence where attention left the page.**

Can EEG and eye tracking determine not only that a reader is mind-wandering, but where the lapse began? Attention Time Machine reconstructs a reader's path through an article, assigns attention probabilities to words or sentences, and creates an "attention bookmark" at the last likely attended point.

The scientific contribution is temporal: the project localizes the onset and recovery of mind-wandering episodes, tests whether inferred attention trajectories predict comprehension, and evaluates whether EEG adds information beyond gaze behavior. A replay interface visualizes gaze, attention probability, episode boundaries, and a simulated rewind policy.

**Dataset:** ROAMM (Reading Observed At Mindless Moments), approximately 50 hours of synchronized 64-channel EEG and eye tracking from 44 participants.

### 2. DREAMSPACE

**An uncertainty-aware EEG system that separates evidence of experience from evidence of recall.**

DREAMSPACE places each awakening on two dimensions: the probability that an experience occurred and the probability that its content remained accessible after waking. This formulation avoids collapsing remembered dreams, "white dreams," and reports of no experience into a single binary label.

The central test asks whether experience-without-recall epochs resemble remembered dreams, no-experience sleep, or neither when projected onto an experience axis trained without them. A console displays each awakening in an experience-recall plane, reports uncertainty and out-of-distribution risk, and simulates performance under reduced EEG montages.

**Dataset:** DREAM (Dream EEG and Mentation database), initially standardized from 20 datasets containing 505 participants and 2,643 awakenings.

### 3. Memory Needle

**A causal EEG system that estimates the instant a cue unlocks a long-term memory.**

Memory Needle processes EEG from a participant's first viewing of a movie clip and determines the earliest moment at which sufficient neural evidence of recognition has accumulated. Recognition-click timestamps from a second viewing provide weak temporal labels for the first-view EEG, avoiding direct mouse-movement leakage during inference.

The project treats recognition as a time-to-event problem, compares familiarity with episodic recollection, and turns the estimated retrieval probability into an adaptive stopping policy. A replay interface commits to a recognition point before revealing the participant's later cue annotation.

**Dataset:** Essex EEG Movie Memory Dataset, containing 64-channel EEG from 27 participants viewing clips from 93 feature films.

### 4. SurpriseGuard

**A perceptual-bandwidth monitor that predicts when surprise may make the next target go unnoticed.**

SurpriseGuard uses only EEG recorded before a probe appears to estimate whether an unexpected sight or sound has temporarily captured enough attention to cause a miss. It separates pre-surprise vulnerability from surprise-evoked capture and tests whether the resulting neural state transfers between vision and hearing.

The engineered system replays held-out trials, displays calibrated miss risk, and simulates whether an alert should be sent, briefly held, or moved to another modality. The project is compact enough for a semester while supporting a strong cross-modal cognitive question.

**Dataset:** Electrophysiological Markers of Surprise-Induced Failures of Visual and Auditory Awareness, containing 60 recordings from 33 participants (approximately 17.7 hours and 0.5 GB).

## Detailed Proposals

## 1. Attention Time Machine

### Research Question

Can a model localize the beginning and end of a mind-wandering episode, and can the inferred attention trajectory predict later comprehension?

Candidate hypotheses include:

- Multimodal EEG and eye tracking localize episode boundaries more accurately than either modality alone.
- EEG provides useful evidence before obvious gaze disruption appears.
- Sustained low-attention probability predicts comprehension failure better than total reading time.
- Limited reader-specific calibration improves generalization.
- Some lapses are primarily ocular, while others show neural evidence without strong gaze changes.

The last hypothesis is particularly useful because it moves the analysis beyond a simple claim that multimodal fusion performs better. Distinct ocular and neural lapse patterns could support a phenotype analysis of mind-wandering episodes.

### Dataset

**ROAMM: Reading Observed At Mindless Moments**

- NEMAR DOI: `10.82901/nemar.on007629`
- OpenNeuro accession: `ds007629`
- Approximately 50 hours of simultaneous 64-channel EEG and eye tracking
- 44 participants completing naturalistic, multi-page reading
- Span-level mind-wandering annotations, fixation data, reading materials, and page-level comprehension scores
- CC0 archive of approximately 22 GB

The benchmark reports leave-one-subject-out detection up to 0.609 AUROC and reduced EEG-to-text decoding during mind-wandering. Preprocessed synchronized derivatives are available and should be used because raw-EEG BIDS conversion remains incomplete.

### ML Design

Reading sessions would be represented at the fixation, word, or short-window level.

| Feature group | Candidate features |
| --- | --- |
| Eye tracking | Fixation duration, refixations, regressions, saccade amplitude, blink rate, pupil measures, local reading speed |
| EEG | Band power, frontal/posterior ratios, spectral entropy, fixation-related potentials, short-range synchrony, change from attentive baseline |
| Context | Word length, lexical frequency, sentence/page position, local text difficulty, elapsed reading time |

Regularized logistic regression and gradient boosting would establish baselines. An HMM or CRF could then smooth window probabilities into coherent attentive and mind-wandering spans. Personalization would compare no calibration, first-page calibration, first-three-page calibration, and within-subject training as an upper bound.

### Engineered System

The replayable reading interface would include:

- The original article and an animated gaze cursor
- An attention-probability trace
- Word- or sentence-level attention shading
- Detected onset and recovery boundaries
- A jump to the last high-confidence attended sentence
- A threshold-and-cooldown intervention policy

A held-out session could be replayed before its retrospective annotation is revealed. Because the dataset is observational, the interface would demonstrate intervention plausibility rather than claim that rewinding improves comprehension.

### Evaluation

Evaluation would include window-level AUROC plus sequence- and interface-relevant metrics:

- Episode-level precision and recall
- Onset error in seconds and words
- Recovery-boundary error
- Episodes detected before their midpoint
- False rewinds per page and average rewind delay
- Fraction of missed content recovered by the bookmark
- Page-level comprehension prediction
- Performance on unseen participants and articles

The primary ablation would compare eye tracking, EEG, fused EEG/eye tracking, behavioral/context features, and all modalities. EEG could remain scientifically informative even if its value is limited to episode onset; a failure to add reliable cross-subject value would also be meaningful under rigorous evaluation.

The published benchmark focuses on present-state detection and the effect of mind-wandering on EEG-to-text decoding. Attention Time Machine instead emphasizes the path into and out of a lapse. Window-level accuracy is therefore secondary to detecting a lapse early enough, placing its onset accurately enough, and recovering a useful portion of the unattended text.

### Scope, Novelty, and Risk

- **Minimum viable:** Reproduce the benchmark on synchronized data and perform modality ablations.
- **Full:** Localize episode boundaries, predict comprehension, and build the replay interface.
- **Stretch:** Cluster lapse phenotypes or add abstention when modalities disagree.

The contribution extends present-state classification through temporal localization, attention-comprehension linkage, modality-disagreement analysis, personalization, and a formally evaluated intervention policy.

The principal risk is the benchmark's modest cross-subject AUROC. Success should therefore mean measurable improvement over eye-only baselines, lower boundary error after temporal smoothing, comprehension prediction beyond reading speed, or benefit from limited calibration—not an unrealistic accuracy target.

## 2. DREAMSPACE

### Research Question

Can EEG separate evidence that an experience occurred from evidence that its content remained accessible after waking?

The standardized labels are:

- **E — Experience:** a specific experience is recalled.
- **EWR — Experience Without Recall:** an experience is reported, but its content is unavailable.
- **NE — No Experience:** no preceding experience is reported.

The proposed representation is:

$$
x = P(\text{experience-like EEG})
$$

$$
y = P(\text{content recalled}\mid\text{reported experience})
$$

The experience axis would be trained on E versus NE while holding out EWR. EWR epochs would then be projected onto that axis. A second axis would distinguish E from EWR.

| Experience evidence | Recall evidence | Interpretation |
| --- | --- | --- |
| Low | Low | No reported experience |
| High | High | Recalled dream |
| High | Low | Candidate white dream |
| Low | High | Inconsistent or out-of-distribution case |

The low-experience/high-recall quadrant is also informative because it exposes awakenings that do not fit the assumed theory instead of forcing every example into a familiar category.

### Dataset

**DREAM: Dream EEG and Mentation database**

- DOI: `10.26180/22133105`
- 20 standardized datasets, 505 participants, and 2,643 awakenings in the initial release
- At least 20 seconds of M/EEG before each awakening
- Six sufficiently compatible datasets yielding 1,462 standardized 30-second epochs in the DREAM paper

An initial pilot could use Tononi Serial Awakenings, which provides 256-channel EEG and repeated awakenings per participant. A published N2 subset contains 261 reports: 110 E, 50 EWR, and 101 NE. The pipeline could then expand one laboratory at a time.

DREAM pools laboratories, procedures, montages, participant populations, awakening protocols, and class distributions. A naive decoder could identify a study, sleep stage, or awakening condition rather than experience. The original DREAM baseline used F4, C4, and O2 with PSD and catch22 features and reported average E-versus-NE AUCs of 0.586 in NREM and 0.700 in REM under random splits. Robustness to grouped and held-out-study evaluation is therefore more important than maximizing a single random-split score.

### ML Design

A low-dimensional common montage using F4, C4, and O2 provides a defensible cross-dataset starting point. Candidate features include band power, spectral slope and entropy, spindles and slow waves, complexity measures, Hjorth parameters, catch22 features, frontal-to-posterior ratios, covariance, synchrony, and probabilistic sleep-stage features.

Candidate models include elastic-net logistic regression, LDA, random forest, histogram gradient boosting, Riemannian tangent-space logistic regression, hierarchical models, isotonic calibration, and calibrated abstention. Sleep stage would serve as a baseline and stratification variable rather than an initial input to the main experience classifier.

The evaluation sequence would become progressively stricter:

1. Random-window validation
2. Awakening-grouped validation
3. Leave-one-participant-out validation
4. Leave-one-dataset or laboratory-out validation
5. Stage-matched evaluation
6. Cross-stage transfer
7. Zero-shot placement of EWR

### Engineered System

The DREAMSPACE console would display:

- Interactive EEG and spectral views
- Sleep-stage probabilities
- Position on the experience-recall plane
- Confidence, calibration, and out-of-distribution warnings
- Influential EEG features
- Similar and dissimilar training laboratories
- Performance under different electrode budgets

A montage-degradation simulator would compare full-density, 32-channel, eight-region, and F4/C4/O2 configurations. The point on the dream plane would move—or the model would abstain—as information is removed.

### Evaluation

Primary metrics would include:

- Mean and worst-dataset AUROC
- Macro balanced accuracy
- Brier score and expected calibration error
- Degradation from random splits to held-out laboratories
- Selective accuracy under abstention
- EWR location relative to E and NE
- EWR consistency across sleep stages and laboratories
- Smallest montage retaining acceptable held-out-laboratory performance

Participant- and dataset-aware confidence intervals would prevent awakenings from being treated as independent when they are clustered within people and studies.

### Scope, Outcomes, and Risk

- **Minimum viable:** Build the Tononi pipeline and test E versus NE with EWR held out.
- **Full:** Harmonize three to six cohorts and perform held-out-laboratory validation.
- **Stretch:** Add montage robustness, calibrated abstention, and cross-stage transfer.

Three outcomes are scientifically useful: EWR may resemble E on experience but differ on recall, appear intermediate, or lose its apparent distinction under laboratory holdout. The last outcome would expose study-specific confounding rather than constitute project failure.

An E-like experience score paired with a lower recall score would support a measurable distinction between experience and post-awakening access. An intermediate position would instead suggest weak or heterogeneous experiences. Either pattern would be evaluated for consistency across stages and laboratories rather than interpreted from pooled examples alone.

The principal operational risk is cross-study harmonization. The deeper limitation is epistemic: labels describe retrospective reports, not direct ground truth about consciousness. All claims should retain the terms "reported experience" and "reported experience without recall."

The underexplored contribution is the combination of a two-axis experience/access formulation, zero-shot EWR placement, held-out-laboratory validation, calibrated abstention, and montage robustness.

## 3. Memory Needle

### Research Question

Does first-view EEG change near the point that a participant later identifies as the recognition-triggering cue, and can that transition distinguish familiarity from episodic recollection?

During a clip's second presentation, the first recognition click supplies a noisy temporal label, $t^*$, which is mapped back to the first presentation:

$$
t^* = \text{candidate memory-triggering cue}
$$

The model would operate causally on first-view EEG and would not receive the second-view click during inference. This reduces the main motor-confound concern.

Three experiments follow:

1. Predict the first replay-click timestamp from first-view EEG, treating never-recognized clips as right-censored.
2. Test whether familiarity and episodic recollection differ in timing or neural signature.
3. Convert retrieval probability into an adaptive policy that stops the clip when $P(\text{retrieval by }t) > \tau$.

The behavioral labels make it possible to test whether familiar recognition emerges earlier or more abruptly than episodic recollection, whether the two forms have different spectral or spatial signatures, and whether stronger reported memories correspond to earlier or more confident neural transitions.

### Dataset

**Essex EEG Movie Memory Dataset**

- OpenNeuro: `ds006142`
- NEMAR: `on006142`
- 64-channel BioSemi EEG sampled at 2,048 Hz
- 27 participants and 10-second clips drawn from 660 clips across 93 films
- Approximately 153–154 clips per participant
- BIDS-formatted raw EEG, synchronized events, survey responses, and CC0 listing on NEMAR

After the first view, clips are labeled not recognized, recognized (movie familiarity), or remembered (scene and plot recollection). Recognized and remembered clips are replayed, during which recognition cues are clicked and rated.

### ML Design

This is a time-to-event problem. At each 100–250 ms step, features would be extracted only from the preceding 500–1,000 ms to estimate a discrete-time hazard:

$$
h(t)=P(T=t\mid T\ge t,\;EEG_{\le t})
$$

Candidate models include discrete-time survival logistic regression, elastic-net Cox regression, random survival forests, gradient-boosted survival trees, HMMs, change-point detection, and calibrated sliding-window boosting.

Feature families could include spectral changes from baseline, band-power slopes, regional summaries, entropy, Hjorth and catch22 features, Riemannian covariance, hemispheric asymmetry, ERP amplitude and slope, and cross-region phase synchrony. A focused comparison among spectral, covariance, and generic time-series representations would keep the study appropriate for an introductory ML course.

The replay click is an uncertain annotation rather than an exact retrieval timestamp. The design would therefore use soft targets, ±500 ms and ±1 second tolerance windows, first/second-view comparisons, tests for systematic click lag, and within-clip timestamp shuffling as a falsification control.

### Engineered System

The Memory Needle player would replay first-view EEG as if it were arriving live and display:

- A clip timeline and retrieval-probability trace
- Current state: no recognition, familiarity, or recollection
- The adaptive stopping point
- The later cue annotation, revealed only after prediction
- Influential frequency bands and regions
- Cumulative cue seconds saved

Policies would include always showing the full clip, stopping at a population median, using behavioral metadata only, using EEG, and using EEG with participant calibration. Copyrighted movie files are not included in the public release, so the initial interface would use timelines and clip identifiers unless access is obtained separately.

### Evaluation

Primary evaluation would hold out an entire participant. A stricter condition would also prevent clips or films from crossing training and test folds.

Metrics would include:

- Median absolute cue-time error
- Event recall within ±0.5 and ±1 second
- Concordance index and time-dependent AUROC
- False triggers on unrecognized clips
- Fraction of each clip shown and cue seconds saved at fixed false-stop rates
- Calibration error
- Performance by recognition type
- Benefit from zero, one, or several calibration clips

Critical baselines include elapsed time, clip type, published memorability, movie-viewing history, shuffled EEG, and shuffled cue timestamps. Since typical clips were marked earlier than neutral clips, EEG must improve over timing and stimulus baselines.

The dataset reports earlier markings for typical clips than neutral clips, with medians of 1.15 and 1.70 seconds. A model could therefore appear successful by exploiting average timing or stimulus class. Subject-and-stimulus holdout is necessary to determine whether the signal reflects retrieval rather than a recognizable dramatic moment, soundtrack, or visual cut.

### Scope, Outcomes, and Risk

- **Minimum viable:** Classify clips as early cue, late cue, or no cue using the first two seconds of first-view EEG.
- **Full:** Predict a continuous event-time distribution and implement the stopping policy.
- **Stretch:** Evaluate participant calibration and double-held-out generalization across people and films.

Success could range from precise cue localization to a reliable early-versus-late distinction. A disappearance of localization under participant-and-film holdout would still show that earlier results depended on stimulus structure.

A strong result would detect a substantial fraction of marked cues at a controlled false-stop rate on unrecognized clips while shortening cue exposure relative to a behavioral-only policy. A more modest result could establish only whether recognition will occur early or late. Both are interpretable if calibration and censoring are handled correctly.

The main limitation is label uncertainty: a replay click is affected by reaction time and repeated exposure and does not identify the exact instant recognition first arose. The project's novelty lies in treating replay clicks as weak first-view labels, applying survival or change-point methods, separating familiarity from recollection in event time, and evaluating a minimal-cue policy.

## 4. SurpriseGuard

### Research Question

Can strictly pre-probe EEG predict whether an unexpected visual or auditory event will cause the next target to be missed? Does the predictive state reflect a modality-general bottleneck or modality-specific sensory disruption?

Surprise-induced blindness is strongest during the first few appearances of a novel, task-irrelevant event and typically affects targets arriving within roughly the next half-second. The auditory analogue appears similarly brief and habituates with repetition, although the auditory evidence should be treated as emerging while its manuscript remains unpublished. The shared paradox is that an event demanding attention may temporarily prevent awareness of what follows.

Each surprise trial would be divided into two causal components:

1. **Pre-surprise vulnerability:** EEG from approximately 500–1,000 ms before surprise, capturing baseline alpha, entropy, variability, and covariance.
2. **Surprise-evoked capture:** EEG after surprise but strictly before probe onset, capturing the neural response to the unexpected event.

The model would estimate:

$$
P(\text{miss}\mid\text{baseline EEG},\;\text{surprise response},\;\text{probe delay})
$$

No EEG sample at or after probe onset would be permitted in the primary model.

### Dataset

**Electrophysiological Markers of Surprise-Induced Failures of Visual and Auditory Awareness**

- OpenNeuro: `ds006394`
- OpenNeuro DOI: `10.18112/openneuro.ds006394.v1.0.3`
- NEMAR: `on006394`
- NEMAR DOI: `10.82901/nemar.on006394`
- 60 recordings from 33 participants, totaling approximately 17.7 hours
- 16 EEG channels at 125 Hz with a left-earlobe reference
- CC0 archive of approximately 0.5 GB

Behavioral files include trial timing, probe position, surprise identity, responses, reaction times, and surprise/probe onsets. Complete task files contain 160 trials, though some participants did not complete both modalities.

Two caveats require explicit handling: the manuscript remains in preparation, and recorded EEG values must be divided by 24 to correct a documented acquisition scaling error.

### ML Design

The compact montage and sampling rate favor robust, low-dimensional features: windowed ERP amplitude, temporal slope and area, theta/alpha/beta power, baseline-relative power change, entropy, Hjorth parameters, regional summaries, covariance, Riemannian features, trial-to-trial response changes, and pre-surprise variability.

Candidate models include elastic-net logistic regression, LDA, linear SVM, random forest, histogram gradient boosting, Riemannian tangent-space logistic regression, and a calibrated behavioral/EEG ensemble.

Three model families would be compared:

- **Behavioral baseline:** probe delay, surprise occurrence, modality, recent performance, task order, reaction-time history, and stimulus identity
- **EEG-only model:** neural information independent of known task structure
- **Fused model:** incremental EEG value beyond timing and trial history

Linear models would provide the main generalization and interpretation tests. Nonlinear models would be retained only if they add reliable held-out-participant value. The most defensible positive result would show that pre-probe EEG improves prediction beyond probe delay, surprise repetition, and recent behavior, rather than merely report a standalone accuracy value.

The strongest experiment is a four-way transfer matrix:

| Training | Testing | Interpretation |
| --- | --- | --- |
| Visual | Visual | Visual miss prediction |
| Auditory | Auditory | Auditory miss prediction |
| Visual | Auditory | Transfer from visual capture to hearing |
| Auditory | Visual | Transfer from auditory capture to vision |

Strong cross-modal transfer would support a supramodal availability state. Strong within-modality performance with poor transfer would support modality-specific mechanisms. Behavioral dominance would indicate that momentary neural vulnerability is weak, individualized, or poorly measured by this montage.

### Engineered System

The Perceptual Bandwidth Monitor would replay trials from a held-out participant and display:

- The task-event stream and scrolling EEG
- Surprise onset
- Baseline-vulnerability and surprise-capture scores
- A calibrated miss-risk estimate
- A send, hold, switch-channel, or abstain decision
- The actual behavioral outcome, revealed after prediction

A configurable policy would delay an alert when:

$$
P(\text{miss}) \times C_{\text{miss}} > C_{\text{delay}}
$$

Policy metrics would include alerts delayed, average delay, impending misses flagged, false holds, misses intercepted at a fixed delay budget, and expected cost relative to always-send and fixed-delay policies. The offline study can establish policy plausibility in simulation, not the real-world benefit of delaying notifications.

The audience-facing replay would commit the risk estimate and policy decision before revealing whether the participant detected the probe. This ordering makes the causal constraint visible and distinguishes the system from a post hoc target-detection decoder.

### Leakage Controls and Evaluation

- **Probe delay:** Compare against a delay-only baseline, match outcomes within delay ranges, and test EEG conditionally on delay.
- **Habituation:** Include surprise occurrence in the baseline and compare trials within similar exposure ranges.
- **Post-target leakage:** Truncate every EEG feature window at the trial's actual probe onset.
- **Participant leakage:** Hold out entire participants in all primary evaluation.
- **Stimulus leakage:** Hold out surprise identities where possible and test sensitivity to identity features.
- **Generic vigilance:** Use ordinary trials to distinguish general inattention from surprise-specific capture.

Nested leave-one-participant-out evaluation would report AUROC, area under the precision-recall curve, balanced accuracy, sensitivity at a fixed false-alarm rate, Brier score, calibration error, incremental EEG value, cross-modal transfer, policy cost, and mean delay. Confidence intervals would bootstrap held-out participants rather than individual trials.

### Scope, Novelty, and Risk

- **Minimum viable:** Correct and preprocess EEG, reproduce behavioral timing/habituation effects, predict visual misses from pre-probe EEG, compare against behavioral baselines, and use participant-level holdout.
- **Full:** Add auditory prediction, cross-modal transfer, vulnerability/capture decomposition, calibration, the replay interface, and the simulated policy.
- **Stretch:** Add participant calibration, recovery curves, susceptibility phenotypes, few-shot cross-modal personalization, and calibrated abstention.

The project's principal risk is the number and distribution of misses per participant, especially if failures cluster in the earliest surprise trials. The unpublished manuscript, incomplete recordings, lack of dedicated EOG channels, and scaling correction add manageable preprocessing and interpretation risks.

Because positive examples may be rare and concentrated by exposure number, ordinary accuracy would be misleading. Precision-recall analysis, calibrated probabilities, participant-level confidence intervals, and within-delay or within-exposure comparisons are required. Conservative preprocessing and frontal-channel sensitivity analyses can partially address the lack of dedicated EOG channels.

The contribution is a causal pre-target prediction framework, a cross-modal test of perceptual availability, explicit behavioral and leakage controls, and a decision-policy evaluation rather than a conventional post-target classifier.
