"""Audit saved sampled production states; never overwrite historical reports."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    trials = [json.loads(line) for line in (args.batch/'trials.jsonl').read_text().splitlines()]
    samples = [json.loads(line) for line in (args.batch/'derived_samples.jsonl').read_text().splitlines()]
    rows = []
    for trial in trials:
        selected = [s for s in samples if s['trial_id'] == trial['trial_id']]
        rows.append(dict(scenario=trial['scenario_id'], sample_count=len(selected),
                         raw_counts=dict(Counter(s['raw_signal_state'] for s in selected)),
                         reported_counts=dict(Counter(s['signal_state'] for s in selected)),
                         changed_samples=sum(s['raw_signal_state'] != s['signal_state'] for s in selected)))
    report = dict(status='historical aggregate cannot certify production behavior',
                  limitation='Derived samples are not all scored frames; no full-frame recall or timing reconstructed.',
                  scenarios=rows)
    output = args.batch/'production-signal-audit.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
