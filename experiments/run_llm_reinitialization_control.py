"""Matched-initial-population mu=0.1 LLM reinitialization control."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from experiments.config.load_env import get_api_key, get_base_url
from experiments.evolution_log import build_evolution_results, trajectory_entry, write_evolution_json
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/llm_reinitialization_20260917'
ARMS = {'v4_1_flash': 'deepseek'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditedPopulation(V2EvolutionaryPopulation):
    """Audit provider responses without changing prompts or offspring operators."""
    def __init__(self, *, audit_path=None, **kwargs):
        super().__init__(**kwargs)
        self.audit_path = audit_path
        self.audit_lock = threading.Lock()
        self.api_success = 0
        self.api_error = 0
        self.audit_proxy = None
        self.job_context = threading.local()

    def _get_llm_client(self):
        client = super()._get_llm_client()
        if self.audit_path is None:
            return client
        with self.audit_lock:
            if self.audit_proxy is None:
                def create(**kwargs):
                    started = time.monotonic()
                    record = {'time_utc': now(), 'generation': self.round_num_offset + 1,
                              'requested_model': kwargs['model'],
                              'operator': getattr(self.job_context, 'operator', None)}
                    try:
                        response = client.chat.completions.create(**kwargs)
                        record.update(ok=True, response_model=response.model,
                                      finish_reason=response.choices[0].finish_reason,
                                      usage=response.usage.model_dump() if response.usage else None)
                        return response
                    except Exception as exc:
                        record.update(ok=False, error_type=type(exc).__name__,
                                      status_code=getattr(exc, 'status_code', None))
                        raise
                    finally:
                        record['elapsed_seconds'] = time.monotonic() - started
                        with self.audit_lock:
                            self.api_success += int(record['ok'])
                            self.api_error += int(not record['ok'])
                            with Path(self.audit_path).open('a', encoding='utf-8') as stream:
                                stream.write(json.dumps(record) + '\n')
                self.audit_proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return self.audit_proxy

    def _run_offspring_job(self, job):
        assert job.operator in ('llm_init', 'llm_mutate'), 'No baseline introductions allowed'
        if job.operator == 'llm_init':
            assert job.parent_code is None and job.parent_id is None
        self.job_context.operator = job.operator
        result = super()._run_offspring_job(job)
        if self.audit_path is not None:
            record = dict(time_utc=now(), generation=job.birth_gen, operator=job.operator,
                          output_index=job.output_index, parent_id=job.parent_id,
                          parent_lineage_id=job.parent_lineage_id, origin=job.origin,
                          code_generated=result.code is not None,
                          code_sha256=hashlib.sha256(result.code.encode()).hexdigest() if result.code else None)
            with self.audit_lock:
                with Path(self.audit_path).with_name('offspring_jobs.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record)+'\n')
        return result

    def _parallel_llm_map(self, fn, jobs):
        jobs = list(jobs)
        before_success, before_error = self.api_success, self.api_error
        results = super()._parallel_llm_map(fn, jobs)
        if jobs and self.api_error > before_error and self.api_success == before_success:
            raise RuntimeError('All API requests in this generation failed; stopping before population commit.')
        return results


def population(cfg, *, provider, audit_path=None, credentials=False):
    if credentials and provider == 'deepseek':
        endpoint = urlsplit(get_base_url(provider))
        if endpoint.scheme != 'https' or endpoint.netloc != 'api.deepseek.com':
            raise ValueError('Official DeepSeek runs require https://api.deepseek.com')
    keys = ('population_size', 'target_interactions_per_gen', 'fitness_window_fraction',
            'benefit', 'cost', 'action_error_probability', 'observation_error_probability',
            'observability', 'observability_p', 'elite_count', 'num_eliminate',
            'tournament_size', 'llm_model', 'mutation_temperature', 'seed',
            'agent_type', 'llm_thinking', 'learning_method', 'fermi_beta',
            'fermi_init_source', 'imitation_learning_mode', 'updates_per_gen', 'llm_concurrency')
    kwargs = {key: cfg[key] for key in keys}
    kwargs.update(mutation_rate_on_adoption=0.1, fermi_init_source="llm", num_generations=cfg['num_generations'],
                  llm_max_tokens_base=cfg['llm_max_tokens'], use_baseline=None,
                  api_key=get_api_key(provider) if credentials else '',
                  api_base_url=get_base_url(provider), audit_path=audit_path)
    return AuditedPopulation(**kwargs)


def evaluate(pop, generation):
    pop._round_offset = generation
    stats = pop._run_one_generation()
    for agent, payoff in zip(pop.agents, stats['payoffs'], strict=True):
        agent.fitness = payoff
    return trajectory_entry(
        generation=generation, cooperation_rate_mean=stats['cooperation_rate_mean'],
        n_interactions=stats['n_interactions'],
        fitness_mean=sum(stats['payoffs']) / len(stats['payoffs']),
        fitness_max=max(stats['payoffs']), population=[pop._agent_record(a) for a in pop.agents])


def checkpoint(pop, trajectory, provenance):
    assert pop.mutation_rate_on_adoption == .1 and pop.fermi_init_source == 'llm'
    assert all(e['parent_lineage_id'] is None for e in pop._lineage_events if e['origin'] == 'independent_init')
    return build_evolution_results(
        trajectory=trajectory, final_population=trajectory[-1]['population'],
        lineage_events=pop._lineage_events,
        config=pop._result_config(len(trajectory), control=provenance,
                                  planned_num_generations=pop.num_generations))


def prepare(out):
    if (out / 'plan.json').exists():
        raise FileExistsError('Plan already exists; use run/status or a new output directory.')
    jobs = []
    for arm, provider in ARMS.items():
        for seed in range(5):
            source = ROOT / f'results/manuscript_v2/{arm}/runs/seed{seed}/evolutionary.json'
            original = read(source)
            cfg = original['config']
            assert cfg['fallback_init_count'] == 0 and cfg['mutation_rate_on_adoption'] == .1
            assert cfg['fermi_init_source'] == 'baseline' and cfg['num_generations'] == 100
            pop = population(cfg, provider=provider)
            initial = original['trajectory'][0]['population']
            pop.agents = [pop._make_agent(a['code'], a['agent_id']) for a in initial]
            pop._next_agent_id = max(a.agent_id for a in pop.agents) + 1
            pop._init_lineage()
            entry = evaluate(pop, 0)
            archived = original['trajectory'][0]
            for key in ('cooperation_rate_mean', 'fitness_mean', 'fitness_max', 'n_interactions'):
                assert abs(entry[key] - archived[key]) < 1e-12, (arm, seed, key, entry[key], archived[key])
            for actual, expected in zip(entry['population'], initial, strict=True):
                assert actual['agent_id'] == expected['agent_id'] and actual['code'] == expected['code']
                assert abs(actual['fitness'] - expected['fitness']) < 1e-12
            provenance = dict(arm=arm, provider=provider, base_url=get_base_url(provider),
                              source=str(source.relative_to(ROOT)), source_sha256=digest(source),
                              intervention='fermi_init_source: baseline -> llm; mutation_rate_on_adoption remains 0.1',
                              initialization='archived generation-0 programs; exact gen0 replay verified',
                              gen0_replay_verified=True, created_utc=now(),
                              planned_num_generations=100,
                              model_identity_note='v4/v4.1 labels follow historical experimenter identification; response aliases logged, service version is not independently fixed for deepseek.')
            folder = out / arm / f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            write_evolution_json(folder / 'initial_checkpoint.json', checkpoint(pop, [entry], provenance))
            save(folder / 'status.json', dict(state='prepared', generation=0, time_utc=now()))
            jobs.append(dict(arm=arm, seed=seed, provider=provider, folder=str(folder), source=str(source)))
            print(f'Gen0 replay verified: {arm} seed{seed}, cooperation={entry["cooperation_rate_mean"]:.5f}', flush=True)
    code_manifest = []
    code_paths = list((ROOT / 'experiments/v2_quantitative').glob('*.py'))
    code_paths += [Path(__file__).resolve(), ROOT / 'experiments/evolution_log.py', ROOT / 'experiments/config/load_env.py']
    for path in code_paths:
        relative = path.relative_to(ROOT)
        copy = out / 'code_snapshot' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        code_manifest.append({'source': relative.as_posix(), 'sha256': digest(path)})
    save(out / 'plan.json', dict(created_utc=now(), jobs=jobs, code_manifest=code_manifest,
                               primary_endpoint='mean cooperation over generations 80..99, per seed',
                               planned_generations=100, workers=5))


def preflight(plan, out):
    import openai
    checks = []
    for provider in dict.fromkeys(j['provider'] for j in plan['jobs']):
        if provider == 'deepseek':
            endpoint = urlsplit(get_base_url(provider))
            if endpoint.scheme != 'https' or endpoint.netloc != 'api.deepseek.com':
                raise ValueError('Official DeepSeek runs require https://api.deepseek.com')
        job = next(j for j in plan['jobs'] if j['provider'] == provider)
        model = read(Path(job['folder']) / 'initial_checkpoint.json')['config']['llm_model']
        if not get_api_key(provider):
            raise RuntimeError(f'Credentials unavailable for {provider}')
        client = openai.OpenAI(api_key=get_api_key(provider), base_url=get_base_url(provider),
                               timeout=60, max_retries=0)
        response = client.chat.completions.create(model=model,
            messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=16,
            temperature=.8, extra_body={'thinking': {'type': 'disabled'}})
        if not response.choices[0].message.content:
            raise RuntimeError(f'Empty preflight response for {provider}')
        checks.append(dict(provider=provider, requested_model=model, response_model=response.model,
                           time_utc=now(), ok=True))
        save(out / 'preflight.json', checks)
        print(f'Provider preflight passed: {provider}, response_model={response.model}', flush=True)


def run_job(job):
    folder = Path(job['folder'])
    with (folder / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            started = time.monotonic()
            try:
                previous = read(folder / ('checkpoint.json' if (folder / 'checkpoint.json').exists() else 'initial_checkpoint.json'))
                cfg = previous['config']
                provenance = cfg['control']
                assert digest(job['source']) == provenance['source_sha256']
                cfg['num_generations'] = provenance['planned_num_generations']
                pop = population(cfg, provider=job['provider'], audit_path=folder / 'api_audit.jsonl', credentials=True)
                last = pop._restore_from_evolution_log(previous)
                pop.rng.setstate(pop._tuple_tree(cfg['rng_state']))
                pop._fallback_mutation_count = cfg['fallback_mutation_count']
                trajectory = previous['trajectory']
                for generation in range(last + 1, 100):
                    pop._round_offset = generation - 1
                    save(folder / 'status.json', dict(state='generating', generation=generation - 1,
                         next_generation=generation, time_utc=now()))
                    pop._select_and_reproduce_by_method(next_gen=generation)
                    trajectory.append(evaluate(pop, generation))
                    result = checkpoint(pop, trajectory, provenance)
                    write_evolution_json(folder / 'checkpoint.json', result)
                    save(folder / 'status.json', dict(state='running', generation=generation,
                         cooperation=trajectory[-1]['cooperation_rate_mean'],
                         fallback_mutation=pop._fallback_mutation_count, time_utc=now()))
                    print(f'Gen {generation}: coop={trajectory[-1]["cooperation_rate_mean"]:.5f}; fallbacks={pop._fallback_mutation_count}', flush=True)
                result = checkpoint(pop, trajectory, provenance)
                write_evolution_json(folder / 'evolutionary.json', result)
                status = dict(state='completed', generation=99, time_utc=now(),
                              elapsed_seconds=time.monotonic()-started,
                              final_cooperation=trajectory[-1]['cooperation_rate_mean'],
                              late20_cooperation=sum(t['cooperation_rate_mean'] for t in trajectory[-20:])/20,
                              fallback_mutation=pop._fallback_mutation_count)
                save(folder / 'status.json', status)
                return dict(arm=job['arm'], seed=job['seed'], **status)
            except Exception as exc:
                state = read(folder / 'status.json')
                state.update(state='failed', error_type=type(exc).__name__, time_utc=now())
                save(folder / 'status.json', state)
                import traceback
                traceback.print_exc()
                return dict(arm=job['arm'], seed=job['seed'], **state)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'status'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == 'prepare':
        prepare(out)
        return
    plan = read(out / 'plan.json')
    if args.mode == 'status':
        for job in plan['jobs']:
            print(job['arm'], job['seed'], json.dumps(read(Path(job['folder']) / 'status.json')))
        return
    lock = out / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        import os
        stream.write(str(os.getpid()))
    try:
        for code in plan['code_manifest']:
            assert digest(ROOT / code['source']) == code['sha256'], f'Code changed: {code["source"]}'
        preflight(plan, out)
        pending = [j for j in plan['jobs'] if read(Path(j['folder']) / 'status.json')['state'] != 'completed']
        summary = []
        with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
            futures = [pool.submit(run_job, job) for job in pending]
            for future in as_completed(futures):
                row = future.result()
                summary.append(row)
                save(out / 'batch_summary.json', summary)
                print(json.dumps(row), flush=True)
        if any(row['state'] != 'completed' for row in summary):
            raise RuntimeError('Some seeds failed; inspect per-seed status/logs.')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
