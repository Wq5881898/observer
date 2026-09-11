"""Grid-search coarse attention constants on approved saved tuning features."""
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import observer.calibration as c
from observer.perception import FeatureVector


BATCHES = (
    Path('data/formal/attention-compensation-postfix/tuning-20260903T071027Z-5fbe7734'),
    Path('data/formal/attention-compensation-postfix/tuning-20260903T071636Z-7243f59f'),
    Path('data/formal/c-closeout/attention-coarse/tuning-20260910T041936Z-c0deec1d'),
    Path('data/formal/c-closeout/attention-coarse/tuning-20260910T043614Z-7eaf4636'),
    Path('data/formal/c-closeout/attention-coarse/tuning-20260910T044434Z-c86dad81'),
    Path('data/formal/c-closeout/attention-coarse/tuning-20260910T050708Z-90a09a7d'),
)
POSITIVE = {'attention_compensation_d03_outside_left', 'attention_compensation_d06_outside_right'}
SCENARIOS = {f'attention_compensation_d0{i}' for i in range(1, 8)}


def main():
    profile = c.load_profile('data/calibration/default.json')
    data = []
    for batch in BATCHES:
        manifest = json.loads((batch/'manifest.json').read_text())
        if manifest['calibration_profile_id'] != profile.profile_id:
            raise ValueError(f'profile mismatch: {batch}')
        baseline = c.AttentionSessionBaseline(**manifest['alignment']['attention_session_baseline'])
        trials = {row['trial_id']: row['scenario_id'] for row in map(json.loads, (batch/'trials.jsonl').read_text().splitlines())}
        for row in map(json.loads, (batch/'derived_samples.jsonl').read_text().splitlines()):
            scenario = trials[row['trial_id']]
            if not any(scenario.startswith(prefix) for prefix in SCENARIOS):
                continue
            feature = FeatureVector('attention', row['attention_smoothed'], reasons=tuple(row['attention_smoothed_reasons']))
            data.append((batch.name, scenario, feature, baseline, scenario in POSITIVE))
    original = (c.ATTENTION_HEAD_YAW_DELTA_THRESHOLD, c.ATTENTION_HEAD_YAW_FUSION_THRESHOLD,
                c.ATTENTION_HEAD_YAW_SCREEN_CORRECTION, c.ATTENTION_COARSE_GAZE_AGREEMENT_MARGIN)
    results = []
    try:
        for head, correction, margin in product(
            (0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30),
            (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
            (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15),
        ):
            c.ATTENTION_HEAD_YAW_DELTA_THRESHOLD = head
            c.ATTENTION_HEAD_YAW_FUSION_THRESHOLD = head
            c.ATTENTION_HEAD_YAW_SCREEN_CORRECTION = correction
            c.ATTENTION_COARSE_GAZE_AGREEMENT_MARGIN = margin
            counts = defaultdict(lambda: [0, 0])
            tp = fp = tn = fn = unknown = 0
            for batch, scenario, feature, baseline, positive in data:
                state = profile.attention.evaluate_with_session_baseline(feature, baseline).state.value
                predicted = state == 'violation'
                counts[(batch, scenario)][1] += 1
                counts[(batch, scenario)][0] += predicted == positive
                unknown += state == 'unknown'
                tp += positive and predicted
                fn += positive and not predicted
                fp += not positive and predicted
                tn += not positive and not predicted
            scenario_scores = [correct/total for correct, total in counts.values()]
            recall = tp/(tp+fn)
            specificity = tn/(tn+fp)
            # Prefer balanced behavior across scenarios and penalize false alerts.
            objective = sum(scenario_scores)/len(scenario_scores) + 0.25*specificity
            results.append(dict(head=head, correction=correction, margin=margin,
                                objective=objective, macro_accuracy=sum(scenario_scores)/len(scenario_scores),
                                worst_scenario=min(scenario_scores), recall=recall, specificity=specificity,
                                unknown_fraction=unknown/len(data), tp=tp, fn=fn, fp=fp, tn=tn))
    finally:
        (c.ATTENTION_HEAD_YAW_DELTA_THRESHOLD, c.ATTENTION_HEAD_YAW_FUSION_THRESHOLD,
         c.ATTENTION_HEAD_YAW_SCREEN_CORRECTION, c.ATTENTION_COARSE_GAZE_AGREEMENT_MARGIN) = original
    results.sort(key=lambda x: (x['objective'], x['worst_scenario'], x['recall']), reverse=True)
    report = {'batches': [str(p) for p in BATCHES], 'sample_count': len(data),
              'selection_note': 'Tuning only; controls weighted through specificity; not holdout.',
              'top_20': results[:20]}
    output = Path('data/formal/c-closeout/attention-coarse/parameter-selection.json')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
