"""Fit conditional velocities on a CPU, with analytic error separation.

This is affine least-squares training in time bins, not neural/image training.
The synthetic bridge is X_t=t Z+(1-t)epsilon, Z~N(2,4), epsilon~N(0,1).
The regression target is V=Z-epsilon. No models or datasets are downloaded.
Running the script writes ch02-results.json beside this file. The optional
--write-figure flag regenerates the analytic variance-path SVG in assets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import argparse
import math
from pathlib import Path
import platform
import random
import statistics
import time


BINS = 32
TRAINING_SAMPLES = 300_000
HELDOUT_SAMPLES = 200_000
GENERATED_SAMPLES = 20_000
EULER_STEPS_PER_BIN = 32
TRAINING_SEED = 184_2026
HELDOUT_SEED = 184_2027
GENERATION_SEED = 184_2028


@dataclass
class Moments:
    count: int = 0
    x: float = 0.0
    v: float = 0.0
    xx: float = 0.0
    xv: float = 0.0
    vv: float = 0.0

    def add(self, x: float, v: float) -> None:
        self.count += 1
        self.x += x
        self.v += v
        self.xx += x*x
        self.xv += x*v
        self.vv += v*v

    def fit(self) -> tuple[float, float]:
        if self.count < 3:
            raise ValueError('Insufficient observations in a time bin')
        mean_x, mean_v = self.x/self.count, self.v/self.count
        variance_x = self.xx/self.count - mean_x*mean_x
        covariance = self.xv/self.count - mean_x*mean_v
        if variance_x <= 0:
            raise ValueError('Degenerate training design')
        slope = covariance/variance_x
        return slope, mean_v-slope*mean_x

    def residual_sum(self, slope: float, intercept: float) -> float:
        return (slope*slope*self.xx + 2*slope*intercept*self.x
                + self.count*intercept*intercept - 2*slope*self.xv
                - 2*intercept*self.v + self.vv)


def oracle_coefficients(t: float) -> tuple[float, float]:
    variance = 5*t*t-2*t+1
    slope = (5*t-1)/variance
    return slope, 2-2*t*slope


def population_bin_coefficients(index: int) -> tuple[float, float]:
    width = 1/BINS
    mean_t = (index+0.5)*width
    variance_t = width*width/12
    variance_x = 5*mean_t*mean_t-2*mean_t+1+9*variance_t
    slope = (5*mean_t-1)/variance_x
    return slope, 2-2*mean_t*slope


def population_risk(coefficients: list[tuple[float, float]]) -> float:
    """Exact risk using time-integrated moments within each bin."""
    risk = 0.0
    for index, (slope, intercept) in enumerate(coefficients):
        mean_t = (index+0.5)/BINS
        mean_tt = mean_t*mean_t+1/(12*BINS*BINS)
        mean_x = 2*mean_t
        mean_xx = 9*mean_tt-2*mean_t+1
        mean_xv = 9*mean_t-1
        risk += (slope*slope*mean_xx + 2*slope*intercept*mean_x
                 + intercept*intercept - 2*slope*mean_xv
                 - 4*intercept + 9)/BINS
    return risk


def exact_piecewise_flow(
        coefficients: list[tuple[float, float]]) -> tuple[float, float]:
    """Return A,B such that the exact time-one solution is A*X_0+B."""
    scale, shift = 1.0, 0.0
    width = 1/BINS
    for slope, intercept in coefficients:
        multiplier = math.exp(slope*width)
        addition = (intercept*math.expm1(slope*width)/slope
                    if abs(slope) > 1e-12 else intercept*width)
        scale, shift = multiplier*scale, multiplier*shift+addition
    return scale, shift


def euler_piecewise_flow(
        coefficients: list[tuple[float, float]],
        steps_per_bin: int = EULER_STEPS_PER_BIN) -> tuple[float, float]:
    scale, shift = 1.0, 0.0
    step = 1/(BINS*steps_per_bin)
    for slope, intercept in coefficients:
        for _ in range(steps_per_bin):
            scale, shift = scale+step*slope*scale, shift+step*(slope*shift+intercept)
    return scale, shift


def euler_oracle_flow(steps: int = BINS*EULER_STEPS_PER_BIN) -> tuple[float, float]:
    scale, shift = 1.0, 0.0
    step = 1/steps
    for index in range(steps):
        slope, intercept = oracle_coefficients(index*step)
        scale, shift = scale+step*slope*scale, shift+step*(slope*shift+intercept)
    return scale, shift


def flow_metrics(flow: tuple[float, float]) -> dict[str, float]:
    scale, shift = flow
    return {'scale': scale, 'mean': shift, 'standard_deviation': abs(scale),
            'variance': scale*scale, 'mean_error': shift-2,
            'standard_deviation_error': abs(scale)-2,
            'gaussian_w2_to_target': math.hypot(shift-2, abs(scale)-2)}


def heldout_checks(coefficients: list[tuple[float, float]]) -> dict:
    rng = random.Random(HELDOUT_SEED)
    model_loss, oracle_loss, differences = [], [], []
    for _ in range(HELDOUT_SAMPLES):
        t = rng.random()
        clean, noise = rng.gauss(2, 2), rng.gauss(0, 1)
        x, target = t*clean+(1-t)*noise, clean-noise
        slope, intercept = coefficients[min(int(t*BINS), BINS-1)]
        oracle_slope, oracle_intercept = oracle_coefficients(t)
        fitted_error = (slope*x+intercept-target)**2
        oracle_error = (oracle_slope*x+oracle_intercept-target)**2
        model_loss.append(fitted_error)
        oracle_loss.append(oracle_error)
        differences.append(fitted_error-oracle_error)
    return {'samples': HELDOUT_SAMPLES,
            'model_cfm_loss': statistics.fmean(model_loss),
            'oracle_cfm_loss': statistics.fmean(oracle_loss),
            'oracle_population_cfm_loss': math.pi,
            'model_loss_standard_error': statistics.stdev(model_loss)/math.sqrt(HELDOUT_SAMPLES),
            'oracle_loss_standard_error': statistics.stdev(oracle_loss)/math.sqrt(HELDOUT_SAMPLES),
            'paired_model_minus_oracle': statistics.fmean(differences),
            'paired_difference_standard_error': statistics.stdev(differences)/math.sqrt(HELDOUT_SAMPLES),
            'difference_note': 'Paired samples reduce comparison noise; the empirical difference may be negative even though population excess risk is nonnegative.'}


def generation_checks(flow: tuple[float, float]) -> dict:
    rng = random.Random(GENERATION_SEED)
    scale, shift = flow
    samples = sorted(scale*rng.gauss(0, 1)+shift for _ in range(GENERATED_SAMPLES))
    target = statistics.NormalDist(2, 2)
    generated_law = statistics.NormalDist(shift, abs(scale))
    def ks_statistic(law: statistics.NormalDist) -> float:
        return max(max((i+1)/GENERATED_SAMPLES-law.cdf(x),
                       law.cdf(x)-i/GENERATED_SAMPLES)
                   for i, x in enumerate(samples))
    return {'samples': GENERATED_SAMPLES,
            'sample_mean': statistics.fmean(samples),
            'sample_variance': statistics.variance(samples),
            'sample_standard_deviation': statistics.stdev(samples),
            'mean_standard_error_under_generated_law': abs(scale)/math.sqrt(GENERATED_SAMPLES),
            'variance_standard_error_under_generated_law': scale*scale*math.sqrt(2/(GENERATED_SAMPLES-1)),
            'ks_to_target': ks_statistic(target),
            'ks_to_generated_exact_law': ks_statistic(generated_law),
            'ks_note': 'Descriptive empirical CDF discrepancies; no p-value or distributional guarantee is inferred.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-figure', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    rng = random.Random(TRAINING_SEED)
    moments = [Moments() for _ in range(BINS)]
    for _ in range(TRAINING_SAMPLES):
        t = rng.random()
        clean, noise = rng.gauss(2, 2), rng.gauss(0, 1)
        x, target = t*clean+(1-t)*noise, clean-noise
        moments[min(int(t*BINS), BINS-1)].add(x, target)
    fitted = [moment.fit() for moment in moments]
    population = [population_bin_coefficients(index) for index in range(BINS)]
    fitted_risk, population_class_risk = population_risk(fitted), population_risk(population)
    fitted_exact = exact_piecewise_flow(fitted)
    population_exact = exact_piecewise_flow(population)
    fitted_euler = euler_piecewise_flow(fitted)
    oracle_euler = euler_oracle_flow()
    decomposition = {
        'representation_and_time_bin_error': [population_exact[0]-2, population_exact[1]-2],
        'finite_training_sample_error': [fitted_exact[i]-population_exact[i] for i in range(2)],
        'euler_solver_error': [fitted_euler[i]-fitted_exact[i] for i in range(2)],
        'total_terminal_error': [fitted_euler[0]-2, fitted_euler[1]-2],
        'coordinate_order': ['scale', 'mean'],
        'note': 'Signed coordinate differences telescope exactly; squared W2 errors are not additive.'}
    for coordinate in range(2):
        reconstructed = sum(decomposition[key][coordinate] for key in
                            ('representation_and_time_bin_error', 'finite_training_sample_error', 'euler_solver_error'))
        if abs(reconstructed-decomposition['total_terminal_error'][coordinate]) > 1e-12:
            raise AssertionError('Terminal error accounting is inconsistent')
    if fitted_risk < population_class_risk-1e-10 or population_class_risk < math.pi-1e-10:
        raise AssertionError('Risk violates the independently derived projection lower bound')
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment': 'Conditional flow regression with fitted affine time bins',
        'model_class': 'One affine scalar function per time bin; no neural network',
        'data': {'clean': 'Z~N(2,4)', 'noise': 'epsilon~N(0,1), independent',
                 'bridge': 'X_t=tZ+(1-t)epsilon', 'target': 'V=Z-epsilon',
                 'time': 'Independent uniform t on [0,1)'},
        'configuration': {'bins': BINS, 'training_samples': TRAINING_SAMPLES,
                          'heldout_samples': HELDOUT_SAMPLES,
                          'generated_samples': GENERATED_SAMPLES,
                          'euler_steps': BINS*EULER_STEPS_PER_BIN,
                          'training_seed': TRAINING_SEED, 'heldout_seed': HELDOUT_SEED,
                          'generation_seed': GENERATION_SEED, 'python': platform.python_version(),
                          'dependencies': 'Python standard library only'},
        'training': {'empirical_cfm_loss': sum(m.residual_sum(*c) for m, c in zip(moments, fitted))/TRAINING_SAMPLES,
                     'bin_counts': [m.count for m in moments],
                     'coefficients': [{'bin': index, 'left': index/BINS, 'right': (index+1)/BINS,
                                       'fitted_slope': pair[0], 'fitted_intercept': pair[1],
                                       'population_slope': population[index][0],
                                       'population_intercept': population[index][1]}
                                      for index, pair in enumerate(fitted)]},
        'population_risk': {'oracle_cfm_loss': math.pi,
                            'population_bin_optimum_cfm_loss': population_class_risk,
                            'fitted_model_cfm_loss': fitted_risk,
                            'time_bin_class_excess': population_class_risk-math.pi,
                            'finite_training_sample_excess': fitted_risk-population_class_risk,
                            'total_fm_loss': fitted_risk-math.pi,
                            'note': 'Exact Gaussian moment integration for the frozen fitted coefficients; not Monte Carlo evaluation.'},
        'heldout': heldout_checks(fitted),
        'terminal': {'target_exact': flow_metrics((2, 2)),
                     'oracle_smooth_euler': flow_metrics(oracle_euler),
                     'population_time_bins_exact_integration': flow_metrics(population_exact),
                     'fitted_time_bins_exact_integration': flow_metrics(fitted_exact),
                     'fitted_time_bins_euler': flow_metrics(fitted_euler),
                     'error_decomposition': decomposition},
        'generated_sample': generation_checks(fitted_euler),
        'solver_refinement': [
            {'steps': BINS*per_bin,
             'oracle_smooth_w2_to_target': flow_metrics(euler_oracle_flow(BINS*per_bin))['gaussian_w2_to_target'],
             'fitted_solver_coordinate_error_norm': math.dist(euler_piecewise_flow(fitted, per_bin), fitted_exact),
             'fitted_total_w2_to_target': flow_metrics(euler_piecewise_flow(fitted, per_bin))['gaussian_w2_to_target']}
            for per_bin in (8, 16, 32, 64)],
        'limitations': ['Synthetic one-dimensional Gaussian target only.',
                        'Affine least squares has an analytic solution; this does not reproduce neural optimization.',
                        'Time binning restricts the fitted function class.',
                        'One training seed is not a study of training variability.',
                        'Population-risk and Gaussian W2 diagnostics rely on this exact synthetic model.',
                        'Terminal first and second moments determine these Gaussian laws, but would not determine a general distribution.']}
    report['runtime_seconds'] = time.perf_counter()-started
    destination = Path(__file__).with_name('ch02-results.json')
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if args.write_figure:
        write_variance_figure()
    print(json.dumps({'configuration': report['configuration'], 'population_risk': report['population_risk'],
                      'heldout': report['heldout'], 'terminal': report['terminal'],
                      'generated_sample': report['generated_sample'],
                      'solver_refinement': report['solver_refinement'],
                      'runtime_seconds': report['runtime_seconds']}, indent=2))


def write_variance_figure() -> None:
    """Compare three analytic probability paths with the same endpoint laws."""
    width, height = 720, 400
    left, top, chart_width, chart_height = 75, 30, 600, 285
    def xy(t, variance):
        return left+chart_width*t, top+chart_height*(1-variance/4.5)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Same Gaussian endpoints, different variance paths</title>',
             '<desc id="desc">Geometric variance 4 to the power t, independent bridge variance 5t squared minus 2t plus 1, and optimal coupling variance (1+t) squared. The independent path first contracts.</desc>',
             '<rect width="720" height="400" fill="#171e26"/>',
             '<g font-family="Georgia,serif" font-size="14" fill="#e3e8ee">']
    for value in range(5):
        _, y = xy(0, value)
        parts.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left+chart_width}" y2="{y:.3f}" stroke="#344251"/>')
        parts.append(f'<text x="{left-16}" y="{y+5:.3f}" text-anchor="end">{value}</text>')
    for value in (0, 0.25, 0.5, 0.75, 1):
        x, y = xy(value, 0)
        parts.append(f'<text x="{x:.3f}" y="{y+25:.3f}" text-anchor="middle">{value:g}</text>')
    parts.extend([f'<text x="{left+chart_width/2}" y="365" text-anchor="middle">Generation time t</text>',
                  '<text x="23" y="175" transform="rotate(-90 23 175)" text-anchor="middle">Variance</text>'])
    paths = [('Chapter 1 geometric', '#b7a8ed', lambda t: 4**t),
             ('Independent bridge', '#88cde7', lambda t: 5*t*t-2*t+1),
             ('Optimal coupling', '#a8dbc2', lambda t: (1+t)**2)]
    for label, color, function in paths:
        points = ' '.join(f'{x:.3f},{y:.3f}' for x, y in
                          (xy(index/200, function(index/200)) for index in range(201)))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for index, (label, color, _) in enumerate(paths):
        x, y = 90+index*200, 391
        parts.append(f'<line x1="{x}" y1="{y-4}" x2="{x+20}" y2="{y-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+27}" y="{y}">{label}</text>')
    parts.append('</g></svg>')
    destination = Path(__file__).resolve().parents[1]/'assets'/'ch02-variance-paths.svg'
    destination.write_text('\n'.join(parts)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
