"""Exact finite-Gaussian checks for Chapter 1; Python standard library only.

No neural network is trained. All values are analytic moments of the declared
ODE/SDE or its discrete scheme. Run with --write to save JSON and the SVG.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def flow_moments(time: float) -> tuple[float, float]:
    """Mean and variance for u(x)=log(2)*(x+2), X0~N(0,1)."""
    growth = 2**time
    return 2*(growth-1), growth**2


def flow_euler_moments(steps: int) -> tuple[float, float]:
    """Exact moments of N explicit Euler updates; raises on invalid steps."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    growth = (1+math.log(2)/steps)**steps
    return 2*(growth-1), growth**2


def ou_moments(time: float, kappa: float, center: float,
               sigma: float, mean0: float, variance0: float):
    """Exact OU moments; initial value is independent of future increments."""
    if time < 0 or kappa <= 0 or sigma < 0 or variance0 < 0:
        raise ValueError('invalid OU parameter')
    decay = math.exp(-kappa*time)
    mean = center+(mean0-center)*decay
    noise_variance = sigma**2*(-math.expm1(-2*kappa*time)) / (2*kappa)
    variance = decay**2*variance0+noise_variance
    return mean, variance


def ou_euler_moments(steps: int, kappa: float, center: float,
                     sigma: float, mean0: float, variance0: float):
    """Propagate Gaussian moments rather than draw noisy estimates."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    mean, variance = mean0, variance0
    h = 1/steps
    factor = 1-kappa*h
    for _ in range(steps):
        mean = center+factor*(mean-center)
        variance = factor**2*variance+sigma**2*h
    return mean, variance


def normal_density(x: float, mean: float, variance: float) -> float:
    return math.exp(-(x-mean)**2/(2*variance))/math.sqrt(2*math.pi*variance)


def figure() -> str:
    """Compute all density-curve coordinates from the Gaussian solution."""
    curves = []
    colors = ['#7dd3fc', '#c4b5fd', '#fbbf24']
    left, top, width, height = 70, 90, 670, 265
    x_min, x_max, y_max = -4.0, 8.0, 0.45
    for time, color in zip([0.0, 0.5, 1.0], colors):
        mean, variance = flow_moments(time)
        points = []
        for index in range(481):
            x = x_min+(x_max-x_min)*index/480
            y = normal_density(x, mean, variance)
            points.append(f'{left+width*(x-x_min)/(x_max-x_min):.3f},'
                          f'{top+height*(1-y/y_max):.3f}')
        curves.append(f'<polyline points="{" ".join(points)}" '
                      f'fill="none" stroke="{color}" stroke-width="2.5"/>')
    ticks = []
    for value in [-4, -2, 0, 2, 4, 6, 8]:
        x = left+width*(value-x_min)/(x_max-x_min)
        ticks.append(f'<text x="{x}" y="380" text-anchor="middle">'
                     f'{value}</text>')
    for value in [0, 0.1, 0.2, 0.3, 0.4]:
        y = top+height*(1-value/y_max)
        ticks.append(f'<text x="56" y="{y+5}" text-anchor="end">'
                     f'{value:g}</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 430" '
            'role="img" aria-labelledby="title desc">'
            '<title id="title">Exact Gaussian probability transport</title>'
            '<desc id="desc">The density moves from mean zero '
            'and variance one to mean two and variance four. '
            'Intermediate time is one half.</desc>'
            '<rect width="800" height="430" fill="#111827"/>'
            '<g fill="#e5e7eb" font-family="Georgia,serif" font-size="17">'
            '<text x="70" y="28">'
            'Exact density under u(x) = log(2)(x + 2)</text>'
            '<text x="70" y="60" fill="#7dd3fc">t = 0</text>'
            '<text x="280" y="60" fill="#c4b5fd">t = 0.5</text>'
            '<text x="500" y="60" fill="#fbbf24">t = 1</text>'
            '<text x="405" y="414" text-anchor="middle">'
            'State x (dimensionless)</text>'
            '<text x="70" y="83">Density</text>'
            + ''.join(ticks) + '</g>'
            f'<path d="M{left},{top}V{top+height}H{left+width}" '
            'stroke="#94a3b8" fill="none"/>' + ''.join(curves) + '</svg>')


def checks():
    expected_mean, expected_variance = flow_moments(1)
    assert expected_mean == 2 and expected_variance == 4
    rows = []
    for steps in [4, 16, 64, 256]:
        mean, variance = flow_euler_moments(steps)
        assert 0 < mean < 2 and 1 < variance < 4
        rows.append({'steps': steps, 'mean': mean, 'variance': variance,
                     'mean_error': expected_mean-mean})
    assert all(rows[i]['mean_error'] > rows[i+1]['mean_error']
               for i in range(len(rows)-1))
    exact_ou = ou_moments(1, 1, 2, math.sqrt(8), 0, 1)
    discrete_ou = ou_euler_moments(4, 1, 2, math.sqrt(8), 0, 1)
    assert abs(discrete_ou[0]-1.3671875) < 1e-12
    assert abs(discrete_ou[1]-4.2138824462890625) < 1e-12
    stationary = ou_moments(1, 1, 0, math.sqrt(2), 0, 1)
    assert abs(stationary[1]-1) < 1e-14
    brownian = [{'steps': n, 'correct_variance': n*(1/n),
                 'incorrect_h_variance': n*(1/n)**2} for n in [4, 16, 64]]
    return {'flow_exact_mean': expected_mean,
            'flow_exact_variance': expected_variance, 'flow_euler': rows,
            'ou_exact_at_one': {'mean': exact_ou[0], 'variance': exact_ou[1]},
            'ou_euler_four_steps': {'mean': discrete_ou[0],
                                    'variance': discrete_ou[1]},
            'ou_euler_stationary_variance_h_quarter': 8/(2-0.25),
            'brownian_scaling': brownian,
            'scope': ('analytic Gaussian moments and discrete '
                      'recurrence checks; no training')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = checks()
    print(json.dumps(result, indent=2))
    if args.write:
        root = Path(__file__).resolve().parents[1]
        (root / 'assets').mkdir(exist_ok=True)
        (root / 'assets' / 'ch01-checks.svg').write_text(
            figure(), encoding='utf-8')
        (root / 'code' / 'ch01-results.json').write_text(
            json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
