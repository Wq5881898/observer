"""Saved-feature diagnosis; not a new holdout or model training."""
import json
import argparse
from collections import Counter
from pathlib import Path
import observer.calibration as c
from observer.perception import FeatureVector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    batch = args.batch
    profile = c.load_profile(Path('data/calibration/default.json'))
    manifest = json.loads((batch/'manifest.json').read_text())
    if profile.profile_id != manifest['calibration_profile_id']:
        raise ValueError('Profile differs from capture')
    baseline = c.AttentionSessionBaseline(**manifest['alignment']['attention_session_baseline'])
    trials = [json.loads(s) for s in (batch/'trials.jsonl').read_text().splitlines()]
    samples = [json.loads(s) for s in (batch/'derived_samples.jsonl').read_text().splitlines()]
    rows = []
    original = c.ATTENTION_HEAD_YAW_FUSION_THRESHOLD
    try:
        for threshold in (0.30, 0.22):
            c.ATTENTION_HEAD_YAW_FUSION_THRESHOLD = threshold
            for trial in trials:
                values = []
                for sample in samples:
                    if sample['trial_id'] != trial['trial_id']:
                        continue
                    features = FeatureVector('attention', sample['attention_smoothed'],
                                             reasons=tuple(sample['attention_smoothed_reasons']))
                    signal = profile.attention.evaluate_with_session_baseline(features, baseline)
                    values.append(dict(time=sample['captured_monotonic'], state=signal.state.value,
                                       original_state=sample['raw_signal_state'], metrics=dict(signal.metrics)))
                rows.append(dict(threshold=threshold, scenario=trial['scenario_id'],
                                 counts=dict(Counter(v['state'] for v in values)), samples=values))
    finally:
        c.ATTENTION_HEAD_YAW_FUSION_THRESHOLD = original
    (batch/'coarse-feature-diagnostic.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k != 'samples'} for r in rows], indent=2))


if __name__ == '__main__':
    main()
