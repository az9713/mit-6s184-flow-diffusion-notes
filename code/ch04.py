"""Exact Gaussian CFG audit; standard library, no training or network access.

Run without arguments for printed results. --write also saves the declared
results and density figure beside this file's portable book directory.
"""
import argparse
import json
import math
from pathlib import Path


def bridge_variances(time):
    if not 0 <= time <= 1:
        raise ValueError('time must be in [0, 1]')
    return (1-time)**2+time**2, (1-time)**2+2*time**2


def velocity_coefficient(time, scale):
    conditional, marginal = bridge_variances(time)
    return (scale*(-2+4*time)/(2*conditional)
            + (1-scale)*(-2+6*time)/(2*marginal))


def exact_variances(time, scale):
    conditional, marginal = bridge_variances(time)
    actual = conditional**scale*marginal**(1-scale)
    precision = scale/conditional+(1-scale)/marginal
    return actual, 1/precision if precision > 0 else None


def euler_variance(steps, scale):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    growth = 1.0
    for n in range(steps):
        growth *= 1+velocity_coefficient(n/steps, scale)/steps
    return growth**2


def gaussian_density(x, variance):
    return math.exp(-x*x/(2*variance))/math.sqrt(2*math.pi*variance)


def density_figure():
    curves = []
    settings = [(1.0, '#7dd3fc'), (0.5, '#fbbf24'), (2/3, '#c4b5fd')]
    for variance, color in settings:
        points = []
        for j in range(481):
            x = -3+6*j/480
            density = gaussian_density(x, variance)
            px, py = 70+660*(x+3)/6, 100+250*(1-density/0.65)
            points.append(f'{px:.3f},{py:.3f}')
        curves.append(f'<polyline points="{" ".join(points)}" '
                      f'fill="none" stroke="{color}" stroke-width="2.5"/>')
    labels = []
    for x in range(-3, 4):
        px = 70+660*(x+3)/6
        labels.append(f'<text x="{px}" y="376" '
                      f'text-anchor="middle">{x}</text>')
    for value in (0, 0.2, 0.4, 0.6):
        py = 100+250*(1-value/0.65)
        labels.append(f'<text x="58" y="{py+5}" '
                      f'text-anchor="end">{value:g}</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 800 425" role="img" '
            'aria-labelledby="title desc">'
            '<title id="title">Exact guidance endpoint counterexample</title>'
            '<desc id="desc">Three centered Gaussian densities: conditional '
            'target variance one, guided ODE variance one half, normalized '
            'density tilt variance two thirds.</desc>'
            '<rect width="800" height="425" fill="#111827"/>'
            '<g fill="#e5e7eb" font-family="Georgia,serif" font-size="16">'
            '<text x="70" y="27">Request y = 0; '
            'total guidance scale w = 2</text>'
            '<text x="70" y="58" fill="#7dd3fc">'
            'Conditional target: variance 1</text>'
            '<text x="430" y="58" fill="#fbbf24">'
            'Guided ODE: variance 1/2</text>'
            '<text x="70" y="85" fill="#c4b5fd">'
            'Normalized tilt: variance 2/3</text>'
            '<text x="70" y="100">Density</text>'
            '<text x="400" y="410" text-anchor="middle">'
            'State x (dimensionless)</text>'
            + ''.join(labels) + '</g>'
            '<path d="M70,110V350H730" stroke="#94a3b8" fill="none"/>'
            + ''.join(curves) + '</svg>')


def checks():
    rows = []
    for scale in (0, 1, 2, 4):
        actual, tilted = exact_variances(1, scale)
        refinement = [{'steps': n, 'variance': euler_variance(n, scale)}
                      for n in (16, 64, 256, 1024)]
        errors = [abs(r['variance']-actual) for r in refinement]
        assert all(a > b for a, b in zip(errors, errors[1:]))
        rows.append({'scale': scale, 'exact_ode_variance': actual,
                     'normalized_tilt_variance': tilted,
                     'euler_refinement': refinement})
    assert exact_variances(1, 2) == (0.5, 2/3)
    for index in range(1, 100):
        time, step = index/100, 1e-6
        derivative = (math.log(exact_variances(time+step, 2)[0])
                      - math.log(exact_variances(time-step, 2)[0]))/(2*step)
        assert abs(derivative-2*velocity_coefficient(time, 2)) < 1e-8
    selected_mean = (-0.9+0.1)/(0.9+0.1)
    assert selected_mean == -0.8
    ordinary_cost, sequential_cost, batched_cost = 40*5+20, 40*10+20, 40*7+20
    return {'guidance': rows, 'biased_dropout_null_mean': selected_mean,
            'worst_case_error_multipliers': [2*w-1 for w in (1, 2, 4)],
            'compatibility_residual_on_ode_variance_at_endpoint_sigma_one':
            1-(2-0.5)*0.5,
            'stipulated_cost_example_ms': {
                'ordinary': ordinary_cost, 'sequential_cfg': sequential_cost,
                'batched_cfg': batched_cost,
                'sequential_ratio': sequential_cost/ordinary_cost,
                'batched_ratio': batched_cost/ordinary_cost},
            'scope': 'exact scalar Gaussian model and deterministic '
            'Euler moments; no neural or perceptual experiment'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = checks()
    print(json.dumps(result, indent=2))
    if args.write:
        root = Path(__file__).resolve().parents[1]
        (root/'code'/'ch04-results.json').write_text(
            json.dumps(result, indent=2)+'\n', encoding='utf-8')
        (root/'assets'/'ch04-guidance.svg').write_text(
            density_figure(), encoding='utf-8')


if __name__ == '__main__':
    main()
