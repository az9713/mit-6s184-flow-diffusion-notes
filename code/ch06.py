"""Finite-state teaching experiments, standard library only.

Run: python code/ch06.py --write-figure
The tabular classifier is fitted on synthetic samples, not a neural model.
"""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
import time


def asymmetric_probability(t):
    """P(X_t=1 | X_0=0) for rates 0->1=2 and 1->0=1."""
    return 2.0 * (1.0 - math.exp(-3.0 * t)) / 3.0


def holding_sample(rng, horizon=1.0):
    state, clock, jumps = 0, 0.0, 0
    while True:
        clock += rng.expovariate(2.0 if state == 0 else 1.0)
        if clock > horizon:
            return state, jumps
        state = 1 - state
        jumps += 1


def uniformized_probability(t, tolerance=1e-15):
    """Poisson mixture; omitted probability mass bounds each entry error."""
    rate = 2.0
    weight = math.exp(-rate * t)
    row = [1.0, 0.0]
    result = [weight, 0.0]
    mass = weight
    n = 0
    # P=I+R/2 = [[0,1],[.5,.5]].
    while 1.0 - mass > tolerance:
        n += 1
        row = [0.5 * row[1], row[0] + 0.5 * row[1]]
        weight *= rate * t / n
        result = [result[k] + weight * row[k] for k in range(2)]
        mass += weight
    return result, max(0.0, 1.0 - mass), n


def path_law(t):
    return {'MM': (1-t)**2,
            '0M': t*(1-t)/2, 'M0': t*(1-t)/2,
            '1M': t*(1-t)/2, 'M1': t*(1-t)/2,
            '00': t*t/2, '11': t*t/2, '01': 0.0, '10': 0.0}


def posterior_zero(state, p_zero=0.5):
    if state == 'MM':
        return p_zero
    if '0' in state and '1' in state:
        raise ValueError('Unidentified posterior at impossible state')
    return 1.0 if '0' in state else 0.0


def generator(states, t):
    rows = {}
    for state in states:
        row = dict.fromkeys(states, 0.0)
        if 'M' in state:
            p = posterior_zero(state)
            for j, token in enumerate(state):
                if token == 'M':
                    for bit, prob in [('0', p), ('1', 1-p)]:
                        dest = state[:j] + bit + state[j+1:]
                        row[dest] += prob / (1-t)
        row[state] = -sum(row.values())
        rows[state] = row
    return rows


def parallel_distribution(steps, p_zero=0.5):
    """Frozen per-token Euler probabilities; exact finite enumeration."""
    law = {'MM': 1.0}
    for k in range(steps):
        reveal = 1.0 / (steps-k)
        following = Counter()
        for state, mass in law.items():
            if 'M' not in state:
                following[state] += mass
                continue
            p = posterior_zero(state, p_zero)
            choices = []
            for token in state:
                choices.append([(token, 1.0)] if token != 'M' else
                               [('M', 1-reveal), ('0', reveal*p),
                                ('1', reveal*(1-p))])
            for pair in itertools.product(*choices):
                dest = ''.join(item[0] for item in pair)
                following[dest] += mass * math.prod(item[1] for item in pair)
        law = dict(following)
    return law


def training_sample(rng):
    bit = '0' if rng.random() < 0.5 else '1'
    t = rng.random()
    state = ''.join(bit if rng.random() < t else 'M' for _ in range(2))
    return state, bit


def fit_table(count, seed):
    rng = random.Random(seed)
    table = {}
    for _ in range(count):
        state, bit = training_sample(rng)
        table.setdefault(state, Counter())[bit] += 1
    fitted = {state: counts['0']/sum(counts.values())
              for state, counts in table.items()}
    return table, fitted


def event_reveal(rng, t, p_zero=0.5):
    """Exact mask-chain law: independent uniform reveal times, shared bit.

    Drawing the bit at the first reveal is equivalent to drawing it at start;
    the second reveal uses the updated posterior and copies that bit.
    """
    reveal_times = [rng.random(), rng.random()]
    bit = '0' if rng.random() < p_zero else '1'
    return ''.join(bit if u <= t else 'M' for u in reveal_times)


def total_variation(left, right):
    return sum(abs(left.get(s, 0)-right.get(s, 0))
               for s in set(left) | set(right)) / 2


def figure(path, parallel):
    labels = ['1', '4', '16', '64']
    values = [parallel[label]['invalid_mass'] for label in labels]
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" '
              'height="390" viewBox="0 0 900 390" role="img" '
              'aria-labelledby="title desc">',
              '<title id="title">Parallel step error in a correlated target</title>',
              '<desc id="desc">Invalid sequences 01 or 10 have probability '
              'one over twice the number of parallel steps. Exact event sampling '
              'has zero invalid mass.</desc>',
              '<rect width="900" height="390" fill="#101827"/>',
              '<g fill="#e2e8f0" font-family="sans-serif">',
              '<text x="45" y="38" font-size="23">Correct token marginals can hide a wrong joint law</text>',
              '<text x="45" y="70" font-size="16">Target: P(00)=P(11)=1/2; invalid: 01 or 10</text>',
              '<path d="M80 110V300H840" stroke="#94a3b8" fill="none"/>']
    for i, (label, value) in enumerate(zip(labels, values)):
        x = 135 + 175*i
        height = value*340
        pieces += [f'<rect x="{x}" y="{300-height}" width="85" '
                   f'height="{height}" fill="#f59e0b"/>',
                   f'<text x="{x+42.5}" y="{285-height}" '
                   f'text-anchor="middle" font-size="16">{value:.6g}</text>',
                   f'<text x="{x+42.5}" y="326" text-anchor="middle" '
                   f'font-size="16">{label} steps</text>']
    pieces += ['<text x="80" y="366" font-size="16">Exact event-time process: invalid mass = 0 at every time.</text>',
               '</g></svg>']
    path.write_text('\n'.join(pieces)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-figure', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    seed = 61842026
    sample_count = 100000
    rng = random.Random(seed)
    outcomes = [holding_sample(rng) for _ in range(sample_count)]
    estimate = sum(x[0] for x in outcomes) / sample_count
    reference = asymmetric_probability(1.0)
    uniformized, tail, terms = uniformized_probability(1.0)
    states = list(path_law(0.5))
    q = path_law(0.37)
    rates = generator(states, 0.37)
    derivative = {'MM': -2*(1-.37),
                  **dict.fromkeys(['0M', 'M0', '1M', 'M1'], (1-2*.37)/2),
                  '00': .37, '11': .37, '01': 0., '10': 0.}
    residual = max(abs(sum(q[x]*rates[x][y] for x in states)-derivative[y])
                   for y in states)
    parallel = {}
    for n in [1, 4, 16, 64]:
        law = parallel_distribution(n)
        invalid = law.get('01', 0)+law.get('10', 0)
        parallel[str(n)] = {'law': law, 'invalid_mass': invalid,
                            'derived_invalid_mass': 1/(2*n),
                            'total_variation': total_variation(law, path_law(1.0))}
    table, fitted = fit_table(80000, seed+1)
    p = fitted['MM']
    rng = random.Random(seed+2)
    losses = []
    for _ in range(sample_count):
        state, bit = training_sample(rng)
        prob = fitted[state] if bit == '0' else 1-fitted[state]
        losses.append(-2*math.log(prob))
    mean_loss = sum(losses)/sample_count
    loss_se = math.sqrt(sum((x-mean_loss)**2 for x in losses)
                        / (sample_count-1)/sample_count)
    rng = random.Random(seed+3)
    empirical = Counter(event_reveal(rng, 0.5) for _ in range(sample_count))
    empirical = {s: n/sample_count for s, n in empirical.items()}
    output = {'scope': 'original finite-state standard-library teaching experiment; no neural benchmark',
              'seed_base': seed, 'monte_carlo_count': sample_count,
              'asymmetric': {'exact_p1': reference, 'holding_time_p1': estimate,
                             'binomial_standard_error': math.sqrt(reference*(1-reference)/sample_count),
                             'mean_jumps': sum(x[1] for x in outcomes)/sample_count,
                             'uniformization_p1': uniformized[1],
                             'omitted_poisson_mass': tail, 'terms': terms,
                             'transpose_derivative_total_mass_at_state0': -1.0},
              'generator_check': {'time': .37, 'max_forward_residual': residual,
                                  'max_row_sum_error': max(abs(sum(row.values())) for row in rates.values())},
              'parallel_exact_enumeration': parallel,
              'event_sampler': {'time': .5, 'empirical_law': empirical,
                                 'total_variation_from_true_path': total_variation(empirical, path_law(.5)),
                                 'invalid_count': round(sample_count*(empirical.get('01', 0)+empirical.get('10', 0)))},
              'tabular_fit': {'training_count': 80000, 'training_seed': seed+1,
                              'counts': table, 'posterior_zero_at_MM': p,
                              'heldout_count': sample_count, 'heldout_seed': seed+2,
                              'heldout_token_summed_nll': mean_loss,
                              'heldout_standard_error': loss_se,
                              'oracle_population_nll': 2*math.log(2)/3,
                              'fitted_population_nll': -(math.log(p)+math.log(1-p))/3,
                              'exact_fitted_terminal_TV': abs(p-.5),
                              'note': 'unrestricted tabular MLE on seven reachable contexts; terminal sampler uses updated posterior after each reveal'},
              'elapsed_seconds': time.perf_counter()-start}
    base = Path(__file__).resolve().parent
    (base/'ch06-results.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    if args.write_figure:
        figure(base.parent/'assets'/'ch06-parallel-error.svg', parallel)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
