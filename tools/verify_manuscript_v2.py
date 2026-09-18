"""Verify archive integrity and manuscript aggregates without rerunning experiments."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
        missing_sources = 0
        changed_sources = 0
        for record in manifest['files']:
            source, copied = ROOT / record['source'], folder / record['copy']
            assert copied.stat().st_size == record['bytes'], copied
            assert sha(copied) == record['sha256'], copied
            if not source.is_file():
                missing_sources += 1
            elif sha(source) != record['sha256']:
                changed_sources += 1
        checks.append(f"{arm}: {len(manifest['files'])} archived files match their recorded SHA-256; original paths unavailable: {missing_sources}; original files changed since archiving: {changed_sources}.")
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

    figure_rows = read_json(ROOT / 'paper_zh/figures/manuscript_v2/fixation_detail_provenance.json')
    fixation_rows = read_csv(ARCHIVE / 'fixation_b3_20260918/fixation_summary.csv')
    assert [row['seed'] for row in figure_rows] == list(range(5))
    for detail in figure_rows:
        assert sha(ROOT / detail['source']) == detail['sha256']
        for index, probe in enumerate(detail['probes']):
            archived = next(row for row in fixation_rows
                            if int(row['seed']) == detail['seed'] and row['probe'] == probe)
            assert detail['forward'][index] == float(archived['rho_candidate_to_probe'])
            assert detail['reverse'][index] == float(archived['rho_probe_to_candidate'])
    checks.append('Figure 5: all 30 directional probabilities match the new b=3 table; five benchmark-source hashes verified.')

    from tools.analyze_paper_fixation_b3 import analyze, tex_probability
    current = analyze()
    assert len(current['rows']) == 50 and current['composition_runs'] == 4750
    assert current['pair_interactions'] == 1900000000
    checks.append('b=3 fixation: five archived candidate hashes, all 100 directional probabilities, 4,750 composition runs, and per-replicate drift checks verified.')

    manuscript = (ROOT / 'paper_zh/manuscript_v2.tex').read_text(encoding='utf-8')
    fixation_table = manuscript.split(r'\label{tab:fixation}', 1)[1].split(r'\end{table*}', 1)[0]
    for seed in range(5):
        values = [value for probe in ('L1', 'ALLC', 'ALLD')
                  for row in current['rows'] if row['seed'] == seed and row['probe'] == probe
                  for value in (row['rho_candidate_to_probe'], row['rho_probe_to_candidate'])]
        expected = f'v4-flash & {seed} & ' + ' & '.join(map(tex_probability, values))
        assert expected in fixation_table, seed
    assert 'sec:scheduling' not in manuscript
    assert 'scheduling_comparison.pdf' not in manuscript
    assert r'\section{Discussion}' not in manuscript
    assert 'b=3' in manuscript.split(r'\label{fig:fixation}', 1)[0].rsplit(r'\caption{', 1)[1]
    checks.append('Table IV: all 30 printed b=3 values agree with the current measurements. Scheduling chapter removed.')
    from tools.analyze_synchronous_evolution import metrics, MATCHED
    evolution = read_json(ARCHIVE / 'synchronous_evolution_20260918/comparison_summary.json')
    assert len(evolution['sources']) == len(evolution['rows']) == 6
    run_data=[]
    for source, expected in zip(evolution['sources'], evolution['rows']):
        assert sha(ROOT / source['path']) == source['sha256']
        data=read_json(ROOT / source['path'])
        assert metrics(data, expected['condition']) == expected
        run_data.append(data)
    for old in run_data[:5]:
        assert all(old['config'][k] == run_data[-1]['config'][k] for k in MATCHED)
    assert run_data[-1]['config']['observation_schedule'] == 'synchronous'
    assert all(old['config']['observation_schedule'] == 'asynchronous' for old in run_data[:5])
    assert evolution['initial_programs_matched'] is False
    assert r'\label{sec:evolution-protocol}' in manuscript
    assert r'\label{fig:evolution-protocol}' in manuscript
    assert f"{evolution['rows'][-1]['late20']:.4f}" in manuscript
    checks.append('Evolution comparison: six source hashes and trajectory summaries verified; one current synchronous run versus five archived asynchronous runs. Original Discussion chapter removed.')
    from tools.analyze_injection_sources import summarize_replays, MATCHED as INJECTION_MATCHED
    from tools.analyze_llm_reinitialization_control import summarize as summarize_injection, validate_jobs
    from collections import Counter
    injection_dir=ARCHIVE/'injection_source_comparison_20260918'
    injection=read_json(injection_dir/'evolution_summary.json')
    operators=Counter()
    computed_probes=[]
    for source in injection['sources']:
        path=ROOT/source['path']; assert sha(path)==source['sha256']
        raw=read_json(path)
        baseline=read_json(ARCHIVE/'v4_1_flash/runs'/f"seed{source['seed']}"/'evolutionary.json')
        assert raw['trajectory'][0]==baseline['trajectory'][0]
        assert all(raw['config'][k]==baseline['config'][k] for k in INJECTION_MATCHED)
        computed,probes,_=summarize_injection(raw,source['condition'],source['seed'])
        computed_probes.extend(probes)
        assert computed==next(r for r in injection['rows'] if r['condition']==source['condition'] and r['seed']==source['seed'])
        if source['condition']=='llm':
            jobs_path=ROOT/source['jobs_path'];assert sha(jobs_path)==source['jobs_sha256']
            jobs=[json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
            operators.update(validate_jobs(raw,jobs))
    assert dict(operators)==injection['operators']=={'llm_init':378,'llm_mutate':3614}
    assert [{k:str(v) for k,v in row.items()} for row in computed_probes]==read_csv(injection_dir/'norm_probes.csv')
    for condition in ('baseline','llm'):
        for generation in (0,99):
            counts=Counter(r['signature'] for r in computed_probes if r['condition']==condition and r['generation']==generation)
            expected=injection['pooled'][condition]['norms'][str(generation)]
            assert dict(counts)==expected['counts']
            assert counts['GGGGBBBB|CDCD']==expected['image_scoring_joint']
            assert sum(n for signature,n in counts.items() if signature.endswith('|CDCD'))==expected['conditional_action']
    for source in read_json(injection_dir/'analysis_manifest.json'):
        assert sha(injection_dir/'source_snapshot'/source['path'])==source['sha256']
    saved_consensus=read_json(injection_dir/'consensus_summary.json')
    assert summarize_replays()==saved_consensus
    assert len(list((injection_dir/'replays').glob('*/*.json')))==550
    assert read_json(injection_dir/'replay_completion.json')['runs']==550
    for path,digest in read_json(ROOT/'paper_zh/figures/manuscript_v2_supplement/injection_comparison_provenance.json').items():
        assert sha(ROOT/path)==digest
    assert r'\label{sec:injection-source}' in manuscript
    assert 'Supplementary Section S3' in manuscript
    assert '0.9333' in manuscript and '0.7383' in manuscript
    checks.append('Injection sources: ten source hashes, exact paired generation-0 records and configured settings, lineage summaries, 378 independent LLM jobs and 3,614 rewrites verified. All 550 replay matrices and nested consensus aggregates verified; fixed replay-seed figure provenance matches.')
    from tools.analyze_worst_fitness_injection import analyze as analyze_worst, OUT as WORST_OUT
    saved_worst=read_json(WORST_OUT/'analysis.json')
    worst=analyze_worst(plot=False)
    assert saved_worst==worst
    assert len(worst['rows'])==10 and len(worst['sources'])==40
    assert worst['provider']=='sili' and worst['llm_concurrency']==4
    assert worst['base_url']=='https://api.siliconflow.cn/v1'
    assert worst['api']['requests']>0
    assert worst['aggregate']['worst_per_generation']['independent_births']['sum']==44
    worst_table=manuscript.split(r'\label{tab:worst-fitness}',1)[1].split(r'\end{table}',1)[0]
    for condition in ('fermi_gated','worst_per_generation'):
        for metric in ('late20','whole_run_mean','final'):
            value=worst['aggregate'][condition][metric]
            formatted=f"${value['mean']:.4f}\\pm{value['sample_sd']:.4f}$".replace('$0.','$.').replace('pm0.','pm.')
            assert formatted in worst_table,(condition,metric,formatted)
    assert r'\label{sec:worst-fitness}' in manuscript and r'\label{fig:worst-fitness}' in manuscript
    assert read_json(ROOT/'paper_zh/figures/manuscript_v2/worst_fitness_provenance.json')['sources']==worst['sources']
    checks.append('Worst-fitness policy chapter: ten completed sili runs with concurrency 4, identical paired initial records and game seeds, 40 source/audit hashes, offspring-parent/code consistency, every injected target and independent trigger, API accounting and printed cooperation means/SDs verified. Withdrawn batches excluded.')
    bibliography = (ROOT / 'paper_zh/manuscript_v2_refs.bib').read_text(encoding='utf-8')
    cited = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', manuscript) for key in group.split(',')}
    entries = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    assert cited == entries and len(entries) == 14
    assert len(re.findall(r'\\begin\{figure\*?\}', manuscript)) == 7
    assert len(re.findall(r'\\begin\{table\*?\}', manuscript)) == 5
    forbidden = r'v4[.\s_\\]+1|two batches|both batches|both models|cross-model'
    assert not re.search(forbidden, manuscript, re.I), 'Supplementary-model content in main TeX'
    supplement = (ROOT / 'paper_zh/manuscript_v2_supplement.tex').read_text(encoding='utf-8')
    assert 'v4.1-flash' in supplement
    assert len(re.findall(r'^v4\.1-flash &', supplement, re.M)) == 5
    assert len(re.findall(r'\\begin\{figure\}', supplement)) == 5
    assert len(re.findall(r'\\begin\{table\}', supplement)) == 4
    assert 'Replacing Canonical Injections with Fresh LLM Programs' in supplement
    for condition in ('baseline','llm'):
        norms=injection['pooled'][condition]['norms']['99']
        assert str(norms['image_scoring_joint'])+'/80' in supplement
        assert str(norms['conditional_action'])+'/80' in supplement
    for stem, document in [('manuscript_v2', manuscript), ('manuscript_v2_supplement', supplement)]:
        for filename in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', document):
            assert (ROOT / 'paper_zh/figures' / stem / filename).is_file(), filename
        log = (WORK / 'build' / f'{stem}.log').read_text(encoding='utf-8', errors='replace')
        assert not re.search(r'Overfull|undefined|Warning|^!', log, re.M), stem
        pdf = WORK / 'build' / f'{stem}.pdf'
        extracted = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-']).decode('utf-8')
        if stem == 'manuscript_v2':
            assert not re.search(forbidden, extracted, re.I), 'Supplementary-model content in main PDF'
        assert '??' not in extracted, stem
        pages = len([page for page in extracted.split('\f') if page.strip()])
        checks.append(f'{stem}: {pages} compiled pages; figure files and cross-references resolve; no warnings or overfull boxes.')
    checks.append('Main paper: 14/14 citation keys, seven figures, five tables; historical supplementary-model label absent from TeX and PDF including figure labels. Current official alias is recorded separately from the primary archived endpoint. Independent worst-fitness replacement chapter added; original Discussion remains removed.')
    checks.append('Supplement: five figures, four tables; matched injection-source experiment with both functional norm probes and reputation agreement, and all five supplementary-model directional fixation rows retained with diagnostic marks.')
    checks.append('Visual and independent citation reviews are recorded separately; this script does not perform those reviews.')
    report = '# Manuscript v2 validation\n\nDate: 2026-09-18\n\n' + '\n'.join('- ' + line for line in checks)
    report += '\n\nScope: archive/data/build checks including the new b=3 fixation benchmarks; this verification script does not rerun simulations. Drift diagnostics are not stationarity proofs; transfer conditions and causal limitations remain explicit in the paper.\n'
    (WORK / 'VALIDATION.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
