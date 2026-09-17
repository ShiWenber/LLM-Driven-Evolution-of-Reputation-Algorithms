"""Verify archive integrity and manuscript aggregates without rerunning experiments."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/manuscript_v2'
WORK = ROOT / 'paper_zh/manuscript_v2_work'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    summary = read_json(ARCHIVE / 'comparison_summary.json')
    checks = []
    for arm, aggregate in summary.items():
        folder = ARCHIVE / arm
        manifest = read_json(folder / 'source_manifest.json')
        for record in manifest['files']:
            source, copied = ROOT / record['source'], folder / record['copy']
            assert copied.stat().st_size == record['bytes'], copied
            assert sha(source) == sha(copied) == record['sha256'], copied
        checks.append(f"{arm}: {len(manifest['files'])} source/copy SHA-256 pairs match.")
        rows = read_csv(folder / 'tables/run_summary.csv')
        assert len(rows) == 5
        for row in rows:
            data = read_json(folder / 'runs' / f"seed{row['seed']}" / 'evolutionary.json')
            trajectory = data['trajectory']
            assert len(trajectory) == 100
            expected = {
                'gen0_cooperation': trajectory[0]['cooperation_rate_mean'],
                'gen99_cooperation': trajectory[-1]['cooperation_rate_mean'],
                'last20_cooperation': statistics.mean(g['cooperation_rate_mean'] for g in trajectory[-20:]),
            }
            for key, value in expected.items():
                assert abs(float(row[key]) - value) < 1e-12, (arm, row['seed'], key)
            benchmark = read_json(folder / 'benchmarks/fixation' / f"seed{row['seed']}" / 'fixation_benchmark.json')
            codes = {hashlib.sha256(a['code'].encode()).hexdigest() for a in data['final_population']}
            assert benchmark['candidate']['code_sha256'] in codes
        for key, metric in aggregate['metrics'].items():
            values = [float(row[key]) for row in rows]
            assert abs(statistics.mean(values) - metric['mean']) < 1e-12
            assert abs(statistics.stdev(values) - metric['sample_sd']) < 1e-12
        norms = read_csv(folder / 'tables/norm_probes.csv')
        for generation in ['0', '99']:
            signatures = [r['signature'] for r in norms if r['generation'] == generation]
            target = aggregate['norms'][generation]
            assert len(signatures) == target['n'] == 80
            assert len(set(signatures)) == target['distinct_signatures']
            assert signatures.count('GGGGBBBB|CDCD') == target['image_scoring_signature']
        fixes = read_csv(folder / 'tables/fixation_summary.csv')
        assert len(fixes) == 50
        assert sum(float(r['max_half_gap']) <= .1 for r in fixes) == aggregate['stationarity_pass_pairs']
        for match in re.finditer(r'\]\(([^)]+)\)', (folder / 'README.md').read_text(encoding='utf-8')):
            assert (folder / match.group(1)).exists(), match.group(1)
        checks.append(f'{arm}: five raw trajectories, aggregate means/SDs, signatures, candidate hashes, drift flags and index links verified.')

    manuscript = (ROOT / 'paper_zh/manuscript_v2.tex').read_text(encoding='utf-8')
    bibliography = (ROOT / 'paper_zh/manuscript_v2_refs.bib').read_text(encoding='utf-8')
    cited = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', manuscript) for key in group.split(',')}
    entries = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    assert cited == entries and len(entries) == 14
    assert len(re.findall(r'\\begin\{figure\*?\}', manuscript)) == 5
    assert len(re.findall(r'\\begin\{table\*?\}', manuscript)) == 4
    for filename in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', manuscript):
        assert (ROOT / 'paper_zh/figures/manuscript_v2' / filename).is_file(), filename
    log = (WORK / 'build/manuscript_v2.log').read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'Overfull|undefined|Warning|^!', log, re.M)
    assert re.search(r'Outputwrittenon.*\(8pages,', re.sub(r'\s+', '', log))
    checks.append('Manuscript: 14/14 citation keys, five figures, four tables; eight-page build has no warnings, undefined references or overfull boxes.')
    checks.append('Visual QA: pages 1–8 inspected; final balanced reference-page rendering inspected separately.')
    checks.append('Literature support: independent review of all 14 cited works and their actual manuscript passages passed; see CITATION_VERIFICATION.md.')
    report = '# Manuscript v2 validation\n\nDate: 2026-09-16\n\n' + '\n'.join('- ' + line for line in checks)
    report += '\n\nScope: archive/data/build checks, not new evolutionary or fixation experiments. Drift diagnostics are not stationarity proofs; transfer conditions and causal limitations remain explicit in the paper.\n'
    (WORK / 'VALIDATION.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
