"""Exact Gaussian VAE and tensor-contract checks; standard library only.

No neural network is trained. Run for JSON stdout; --write also saves the
result and an original SVG of the likelihood/variational-objective curves.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


A = (1.0, 2.0)
OBSERVATION = (2.0, 1.0)


def gaussian_kl(mean_q, variance_q, mean_p=0.0, variance_p=1.0):
    """KL of two scalar Gaussian laws with strictly positive variances."""
    if variance_q <= 0 or variance_p <= 0:
        raise ValueError('Gaussian density variances must be positive')
    return 0.5*(math.log(variance_p/variance_q)
                + (variance_q+(mean_q-mean_p)**2)/variance_p-1)


def posterior(x=OBSERVATION, decoder_variance=1.0, rho=1.0):
    """Optimal Gaussian encoder for a fixed decoder and positive KL weight."""
    if decoder_variance <= 0 or rho <= 0:
        raise ValueError('variance and KL weight must be positive')
    denominator = sum(a*a for a in A)+rho*decoder_variance
    return (sum(a*y for a, y in zip(A, x))/denominator,
            rho*decoder_variance/denominator)


def reconstruction_nll(mean, variance, decoder_variance=1.0):
    if variance <= 0 or decoder_variance <= 0:
        raise ValueError('variances must be positive')
    residual = sum((y-a*mean)**2+a*a*variance
                   for a, y in zip(A, OBSERVATION))
    return (math.log(2*math.pi*decoder_variance)
            + residual/(2*decoder_variance))


def negative_objective(mean, log_variance, rho=1.0):
    variance = math.exp(log_variance)
    return reconstruction_nll(mean, variance)+rho*gaussian_kl(mean, variance)


def log_evidence():
    """Log N(x;0,[[2,2],[2,5]]) at x=(2,1)."""
    x0, x1 = OBSERVATION
    quadratic = (5*x0*x0-4*x0*x1+2*x1*x1)/6
    return -math.log(2*math.pi)-0.5*math.log(6)-0.5*quadratic


def quadrature_probe(mean, log_variance):
    """Normal quadrature exact for the quadratic and its derivatives."""
    sd = math.exp(log_variance/2)
    total = grad_mean = grad_log_variance = 0.0
    nodes = ((-math.sqrt(3), 1/6), (0.0, 2/3), (math.sqrt(3), 1/6))
    for noise, weight in nodes:
        latent = mean+sd*noise
        residual = [a*latent-y for a, y in zip(A, OBSERVATION)]
        derivative = sum(a*r for a, r in zip(A, residual))
        total += weight*(math.log(2*math.pi)
                         + sum(r*r for r in residual)/2)
        grad_mean += weight*derivative
        grad_log_variance += weight*derivative*sd*noise/2
    variance = sd*sd
    return (total+gaussian_kl(mean, variance),
            [grad_mean+mean, grad_log_variance+(variance-1)/2])


def central_gradient(mean, log_variance, step=1e-5):
    return [
        (negative_objective(mean+step, log_variance)
         - negative_objective(mean-step, log_variance))/(2*step),
        (negative_objective(mean, log_variance+step)
         - negative_objective(mean, log_variance-step))/(2*step)]


def patchify(values, channels, height, width, patch):
    """Channel-first input -> spatial-major patches, channel-major."""
    if (patch <= 0 or height % patch or width % patch
            or len(values) != channels*height*width):
        raise ValueError('input shape must match and patch must divide sizes')
    return [[values[(c*height+i+di)*width+j+dj]
             for c in range(channels)
             for di in range(patch) for dj in range(patch)]
            for i in range(0, height, patch)
            for j in range(0, width, patch)]


def unpatchify(tokens, channels, height, width, patch):
    if (patch <= 0 or height % patch or width % patch
            or len(tokens) != (height//patch)*(width//patch)
            or any(len(row) != channels*patch*patch for row in tokens)):
        raise ValueError('token layout is incompatible with output shape')
    values = [None]*(channels*height*width)
    for index, row in enumerate(tokens):
        i = (index//(width//patch))*patch
        j = (index % (width//patch))*patch
        for c in range(channels):
            for di in range(patch):
                for dj in range(patch):
                    values[(c*height+i+di)*width+j+dj] = (
                        row[(c*patch+di)*patch+dj])
    return values


def attention(queries, keys, values):
    """Tiny row-softmax scaled attention, without a trained projection."""
    if not queries or not queries[0]:
        raise ValueError('queries must be a nonempty rectangular array')
    width = len(queries[0])
    if (not keys or len(keys) != len(values)
            or any(len(row) != width for row in queries+keys)
            or any(len(row) != len(values[0]) for row in values)):
        raise ValueError('incompatible attention shapes')
    output, probabilities = [], []
    for query in queries:
        logits = [sum(q*k for q, k in zip(query, key))/math.sqrt(width)
                  for key in keys]
        exponentials = [math.exp(x-max(logits)) for x in logits]
        weights = [x/sum(exponentials) for x in exponentials]
        probabilities.append(weights)
        output.append([sum(w*row[j] for w, row in zip(weights, values))
                       for j in range(len(values[0]))])
    return output, probabilities


def run_checks():
    evidence = log_evidence()
    true_mean, true_variance = posterior()
    rows = []
    encoder_choices = [(0.5, 0.25), (true_mean, true_variance), (0, 1)]
    for mean, variance in encoder_choices:
        recon = reconstruction_nll(mean, variance)
        prior_kl = gaussian_kl(mean, variance)
        gap = gaussian_kl(mean, variance, true_mean, true_variance)
        elbo = -recon-prior_kl
        assert abs(evidence-elbo-gap) < 1e-12
        rows.append(dict(mean=mean, variance=variance,
                         reconstruction_nll=recon,
                         prior_kl=prior_kl, elbo=elbo, posterior_kl=gap))
    mean, ell = 0.5, math.log(0.25)
    quadrature_value, quadrature_gradient = quadrature_probe(mean, ell)
    analytic_gradient = [5*mean-4+mean,
                         5*math.exp(ell)/2+(math.exp(ell)-1)/2]
    finite_gradient = central_gradient(mean, ell)
    assert abs(quadrature_value-negative_objective(mean, ell)) < 1e-12
    assert max(abs(a-b) for a, b in zip(
        analytic_gradient, quadrature_gradient)) < 1e-12
    assert max(abs(a-b) for a, b in zip(
        analytic_gradient, finite_gradient)) < 1e-8
    weighted = []
    for rho in [0.5, 1, 2, 10]:
        m, v = posterior(rho=rho)
        assert abs(5*m-4+rho*m) < 1e-12
        assert abs(5/2+rho*(1-1/v)/2) < 1e-12
        value = -negative_objective(m, math.log(v), rho)
        weighted.append(dict(rho=rho, mean=m, variance=v, objective=value))
    sigma = [[2, 2], [2, 5]]
    mean_variance = sum(A[i]*sigma[i][j]*A[j]
                        for i in range(2) for j in range(2))/36
    aggregate_variance = mean_variance+true_variance
    assert abs(aggregate_variance-1) < 1e-12
    mutual_information = 0.5*math.log(6)
    variance_sum = mean_variance+true_variance
    expected_prior_kl = 0.5*(variance_sum-math.log(true_variance)-1)
    assert abs(mutual_information-expected_prior_kl) < 1e-12
    residual_map = [[int(i == j)-A[i]*A[j]/6
                     for j in range(2)] for i in range(2)]
    mean_mse = sum(residual_map[k][i]*sigma[i][j]*residual_map[k][j]
                   for k in range(2) for i in range(2) for j in range(2))
    sampled_mse = mean_mse+sum(a*a for a in A)*true_variance
    decoded_mse = sampled_mse+2
    assert abs(mean_mse-7/6) < 1e-12
    assert abs(sampled_mse-2) < 1e-12
    assert abs(decoded_mse-4) < 1e-12
    generated_cov = [[A[i]*A[j]+int(i == j)
                      for j in range(2)] for i in range(2)]
    assert generated_cov == sigma
    data = list(range(2*4*6))
    tokens = patchify(data, 2, 4, 6, 2)
    recovered = unpatchify(tokens, 2, 4, 6, 2)
    assert recovered == data
    queries = [[1, 0], [0, 1], [1, 1]]
    keys = [[1, 0], [0, 1]]
    values = [[2, 0, -1], [0, 4, 1]]
    output, weights = attention(queries, keys, values)
    assert len(output) == 3 and len(output[0]) == 3
    assert max(abs(sum(row)-1) for row in weights) < 1e-12
    assert output[-1] == [1, 2, 0]
    return dict(
        model=dict(A=A, x=OBSERVATION, decoder_variance=1,
                   covariance=sigma, log_evidence=evidence),
        encoders=rows, weighted_optima=weighted,
        gradient=dict(analytic=analytic_gradient,
                      quadrature=quadrature_gradient,
                      finite_difference=finite_gradient,
                      max_fd_error=max(abs(a-b) for a, b in zip(
                          analytic_gradient, finite_gradient))),
        latent_laws=dict(encoder_mean_variance=mean_variance,
                         sampled_aggregate_variance=aggregate_variance,
                         expected_encoder_prior_kl=expected_prior_kl,
                         mutual_information=mutual_information,
                         deterministic_decoder_covariance=[[1, 2], [2, 4]],
                         sampled_decoder_covariance=generated_cov,
                         mean_encoder_decoder_mse=mean_mse,
                         sampled_encoder_mean_decoder_mse=sampled_mse,
                         sampled_encoder_sampled_decoder_mse=decoded_mse),
        dimensions=dict(pixel_scalars=3*512**2, latent_scalars=4*64**2,
                        scalar_reduction=(3*512**2)/(4*64**2),
                        latent_tokens=32**2, pixel_tokens=256**2,
                        self_attention_pairs=1024**2,
                        cross_attention_pairs=1024*12,
                        velocity_output_patch_width=4*2**2,
                        original_dit_output_patch_width=8*2**2),
        video_contract=dict(raw_frames=17, latent_frames=math.ceil(17/8),
                            latent_tokens=math.ceil(17/8)*32**2,
                            patch_width=16*2**2, decoded_frames=24,
                            scalar_ratio=(17*3*512**2)/(3*16*64**2)),
        patch_round_trip=dict(shape=[2, 4, 6], tokens=len(tokens),
                              patch_width=len(tokens[0]), passed=True),
        attention=dict(output=output, probabilities=weights),
        scope='Exact Gaussian formulas and finite CPU contracts; no training.')


def make_figure(path):
    """Original computed curves; neither a paper figure nor fitted output."""
    evidence = log_evidence()
    variance = 1/6
    xmin, xmax, ymin, ymax = -0.2, 1.5, -7.0, -3.1
    px = lambda x: 76+760*(x-xmin)/(xmax-xmin)
    py = lambda y: 330-260*(y-ymin)/(ymax-ymin)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 430" '
             'role="img" aria-labelledby="title desc">',
             '<title id="title">Likelihood and variational objectives</title>',
             '<desc id="desc">Fixed encoder variance one sixth. The ordinary '
             'ELBO meets exact log likelihood at mean two thirds. '
             'A half-weight '
             'KL objective can exceed log likelihood.</desc>',
             '<rect width="900" height="430" fill="#101824"/>',
             '<g fill="#dce7f1" font-family="sans-serif" font-size="15">',
             '<text x="76" y="30">A likelihood bound depends on '
             'its KL weight</text>']
    for value in [-7, -6, -5, -4]:
        y = py(value)
        parts += [f'<path d="M76 {y}H836" stroke="#334457"/>',
                  f'<text x="40" y="{y+5}">{value}</text>']
    for value in [0, 0.5, 1, 1.5]:
        x = px(value)
        parts.append(f'<text x="{x-8}" y="355">{value}</text>')
    parts.append('<text x="380" y="384">encoder mean m</text>')
    colors = [(1, '#65d6cf'), (0.5, '#f4b860')]
    for rho, color in colors:
        points = []
        for i in range(201):
            m = xmin+(xmax-xmin)*i/200
            objective = -negative_objective(m, math.log(variance), rho)
            points.append(f'{px(m):.3f},{py(objective):.3f}')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" '
                     f'stroke="{color}" stroke-width="3"/>')
    parts.append(f'<path d="M76 {py(evidence)}H836" stroke="#c0b6ff" '
                 'stroke-width="2" stroke-dasharray="7 5"/>')
    parts += ['<text x="76" y="413" fill="#65d6cf">ELBO (rho=1)</text>',
              '<text x="300" y="413" fill="#f4b860">weighted (rho=0.5)</text>',
              '<text x="570" y="413" fill="#c0b6ff">exact log '
              'likelihood</text>',
              '</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    results = run_checks()
    output = json.dumps(results, indent=2, sort_keys=True)+'\n'
    if args.write:
        here = Path(__file__).resolve().parent
        (here/'ch05-results.json').write_text(output, encoding='utf8')
        make_figure(here.parent/'assets'/'ch05-elbo.svg')
    print(output, end='')


if __name__ == '__main__':
    main()
