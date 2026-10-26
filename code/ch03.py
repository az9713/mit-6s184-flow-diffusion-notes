"""Chapter 3 analytic checks, using only Python's standard library.

The Gaussian bridge is X_t=t*Z+(1-t)*epsilon, Z~N(2,4), epsilon~N(0,1).
Checks concern exact scores/moments and finite-step recurrences, not training.
Run without options for JSON stdout; --write also saves results and one SVG.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def gaussian(time):
    """Return the bridge's exact marginal mean and variance."""
    return 2*time, 4*time**2+(1-time)**2


def score(time, x):
    mean, variance = gaussian(time)
    return -(x-mean)/variance


def velocity(time, x):
    mean, variance = gaussian(time)
    return 2+(5*time-1)*(x-mean)/variance


def posterior_mean(time, x):
    mean, variance = gaussian(time)
    return 2+4*time*(x-mean)/variance


def reciprocal_variance_integral(time):
    """Exact integral of 1/(5*t*t-2*t+1) from zero to time."""
    return (math.atan((5*time-1)/2)+math.atan(0.5))/2


def exact_variance(time, sigma, correction):
    """Variance under b=u+correction*sigma^2*s/2; correction=1 is right."""
    _, target = gaussian(time)
    integral = reciprocal_variance_integral(time)
    if correction == 1:
        return target
    if correction == 0:
        return target*(1+sigma**2*integral)
    if correction == -1:
        return target*(2*math.exp(sigma**2*integral)-1)
    raise ValueError('correction must be -1, 0, or 1')


def euler_path(steps, sigma, correction):
    """Exact Gaussian moments of the Euler–Maruyama recurrence."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if sigma < 0 or correction not in [-1, 0, 1]:
        raise ValueError('invalid diffusion amplitude or correction')
    h = 1/steps
    mean, variance = 0.0, 1.0
    rows = [(0.0, mean, variance)]
    for n in range(steps):
        time = n*h
        _, target_variance = gaussian(time)
        coefficient = ((10*time-2)-correction*sigma**2)/(2*target_variance)
        mean += h*(2+coefficient*(mean-2*time))
        variance = (1+h*coefficient)**2*variance+sigma**2*h
        rows.append(((n+1)*h, mean, variance))
    return rows


def normal_density(x, mean, variance):
    return math.exp(-(x-mean)**2/(2*variance))/math.sqrt(2*math.pi*variance)


def mixture(time, x):
    """A separate synthetic clean mixture: 0.5*N(-2,1/4)+0.5*N(2,1/4)."""
    component_mean = 2*time
    variance = time**2/4+(1-time)**2
    left = normal_density(x, -component_mean, variance)
    right = normal_density(x, component_mean, variance)
    density = (left+right)/2
    derivative = (-(x+component_mean)*left-(x-component_mean)*right)
    derivative /= 2*variance
    return density, derivative/density


def figure():
    """Plot coordinates come from exact formulas or the EM recurrence."""
    x0, y0, width, height = 70, 95, 660, 270

    def points(rows):
        return ' '.join(f'{x0+width*t:.3f},{y0+height*(1-v/8):.3f}'
                        for t, v in rows)

    exact = [(i/400, gaussian(i/400)[1]) for i in range(401)]
    uncorrected = [(i/400, exact_variance(i/400, 1, 0))
                   for i in range(401)]
    discrete = [(t, v) for t, _, v in euler_path(32, 1, 1)]
    axes = []
    for t in [0, 0.25, 0.5, 0.75, 1]:
        axes.append(f'<text x="{x0+width*t}" y="390" '
                    f'text-anchor="middle">{t:g}</text>')
    for value in [0, 2, 4, 6, 8]:
        axes.append(f'<text x="55" y="{y0+height*(1-value/8)+5}" '
                    f'text-anchor="end">{value}</text>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 440" '
        'role="img" aria-labelledby="title desc">'
        '<title id="title">'
        'Score correction preserves Gaussian variance</title>'
        '<desc id="desc">Target variance and corrected exact SDE coincide. '
        'An uncorrected SDE has final variance four plus pi. A 32-step '
        'corrected recurrence retains a smaller numerical bias.</desc>'
        '<rect width="800" height="440" fill="#111827"/>'
        '<g fill="#e5e7eb" font-family="Georgia,serif" font-size="16">'
        '<text x="70" y="27">'
        'Same Gaussian path, diffusion amplitude σ = 1</text>'
        '<text x="70" y="57" fill="#fbbf24">'
        'Target / exact corrected SDE</text>'
        '<text x="430" y="57" fill="#fb7185">Noise without correction</text>'
        '<text x="70" y="81" fill="#7dd3fc">Corrected EM, 32 steps</text>'
        '<text x="405" y="426" text-anchor="middle">Generation time t</text>'
        '<text x="14" y="91">Variance</text>'+''.join(axes)+'</g>'
        f'<path d="M{x0},{y0}V{y0+height}H{x0+width}" '
        'stroke="#94a3b8" fill="none"/>'
        f'<polyline points="{points(exact)}" stroke="#fbbf24" '
        'stroke-width="3" fill="none"/>'
        f'<polyline points="{points(uncorrected)}" stroke="#fb7185" '
        'stroke-width="2.5" fill="none"/>'
        f'<polyline points="{points(discrete)}" stroke="#7dd3fc" '
        'stroke-width="2.5" stroke-dasharray="7 5" fill="none"/></svg>'
    )


def checks():
    result = {'gaussian_scores': [], 'samplers': []}
    for time in [0.25, 0.5, 0.9, 0.99]:
        mean, variance = gaussian(time)
        for x in [-1.0, 0.0, 2.0]:
            posterior_score = -(x-time*posterior_mean(time, x))/(1-time)**2
            assert abs(posterior_score-score(time, x)) < 1e-10
            reconstructed = (x+(1-time)*score(time, x))/time
            assert abs(reconstructed-velocity(time, x)) < 1e-12
        floor = 1/(1-time)**2-1/variance
        result['gaussian_scores'].append(
            {'t': time, 'mean': mean, 'variance': variance,
             'DSM_floor': floor, 'noise_MSE_floor': (1-time)**2*floor})
    for sigma in [0.0, 0.5, 1.0, 2.0]:
        rows = []
        for steps in [32, 128, 512, 2048]:
            _, mean, variance = euler_path(steps, sigma, 1)[-1]
            rows.append({'steps': steps, 'mean': mean, 'variance': variance,
                         'absolute_variance_error': abs(variance-4)})
        assert rows[-1]['absolute_variance_error'] < rows[0][
            'absolute_variance_error']
        result['samplers'].append(
            {'sigma': sigma, 'corrected_EM': rows,
             'uncorrected_exact_endpoint': exact_variance(1, sigma, 0),
             'wrong_sign_exact_endpoint': exact_variance(1, sigma, -1)})
    time, x, h = 0.5, 0.7, 1e-5
    density, analytic = mixture(time, x)
    difference = (math.log(mixture(time, x+h)[0]) -
                  math.log(mixture(time, x-h)[0]))/(2*h)
    assert abs(analytic-difference) < 1e-8
    component_variance = time**2/4+(1-time)**2
    curvature = (2*time)**2/component_variance**2-1/component_variance
    assert mixture(time, 0)[1] == 0 and curvature > 0
    assert abs(reciprocal_variance_integral(1)-math.pi/4) < 1e-14
    result['mixture'] = {'time': time, 'x': x, 'density': density,
                         'analytic_score': analytic,
                         'finite_difference_score': difference,
                         'score_at_zero': 0,
                         'density_curvature_ratio': curvature}
    times = [0.25, 0.75]
    variances = [gaussian(t)[1] for t in times]
    weights = [(1-t)**2 for t in times]
    unweighted_optimum = 2/sum(variances)
    weighted_optimum = sum(weights)/sum(
        weight*variance for weight, variance in zip(weights, variances))
    assert abs(unweighted_optimum-16/25) < 1e-14
    assert abs(weighted_optimum-80/77) < 1e-14
    result['restricted_weighting'] = {
        'times': times, 'variances': variances,
        'unweighted_shared_precision_optimum': unweighted_optimum,
        'noise_weights': weights,
        'noise_weighted_shared_precision_optimum': weighted_optimum}
    result['scope'] = ('analytic Gaussian and mixture checks; '
                       'no neural training')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = checks()
    print(json.dumps(result, indent=2))
    if args.write:
        root = Path(__file__).resolve().parents[1]
        (root / 'assets').mkdir(exist_ok=True)
        (root / 'assets' / 'ch03-variance.svg').write_text(
            figure(), encoding='utf-8')
        (root / 'code' / 'ch03-results.json').write_text(
            json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
