"""Execute declared standalone listings and programs in a portable book copy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = {'created_at', 'elapsed_seconds', 'runtime_seconds', 'elapsed_s', 'runtime_s',
            'wall_time_seconds'}


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapters', nargs='*', type=int)
    args = parser.parse_args()
    numbers = args.chapters or [int(p.stem[2:])
                               for p in (ROOT/'code').glob('ch[0-9][0-9].py')]
    checks, failures = [], []
    temporary = tempfile.TemporaryDirectory(prefix='6s184-code-')
    work = Path(temporary.name).resolve()
    expected_parent = Path(tempfile.gettempdir()).resolve()
    if not work.is_relative_to(expected_parent) or not work.name.startswith('6s184-code-'):
        raise ValueError('Unexpected temporary execution directory')
    try:
        isolated_code = work/'book'/'code'
        shutil.copytree(ROOT/'code', isolated_code,
                        ignore=shutil.ignore_patterns('*-results.json', '__pycache__', '*.pyc'))
        (work/'book'/'assets').mkdir()
        for number in sorted(numbers):
            ident = f'ch{number:02d}'
            source = ROOT/'chapters'/(ident+'.html')
            soup = BeautifulSoup(source.read_text(encoding='utf-8'), 'html.parser')
            for i, code in enumerate(soup.select('pre code'), 1):
                classes = ' '.join(code.get('class', []))
                if classes and 'python' not in classes:
                    continue
                script = work/f'{ident}-listing-{i}.py'
                script.write_text(code.get_text()+'\n', encoding='utf-8')
                process = subprocess.run([sys.executable, str(script)], cwd=work,
                                         capture_output=True, text=True,
                                         encoding='utf-8', timeout=60)
                expected = None
                wrapper = code.find_parent('div', class_='codewrap') or code.parent
                following = wrapper.find_next_sibling()
                if following and following.name == 'pre' and 'output' in following.get('class', []):
                    expected = following.get_text().strip()
                matches = expected is None or process.stdout.strip() == expected
                record = {'chapter': number, 'listing': i,
                          'exit_code': process.returncode,
                          'displayed_output_checked': expected is not None,
                          'displayed_output_matches': matches,
                          'code_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}
                checks.append(record)
                if process.returncode or not matches:
                    failures.append(f'{ident} listing {i}: execution/output mismatch')
            program = ROOT/'code'/(ident+'.py')
            copied_program = isolated_code/program.name
            process = subprocess.run([sys.executable, str(copied_program)], cwd=work,
                                     capture_output=True, text=True,
                                     encoding='utf-8', timeout=60)
            result_path = program.with_name(ident+'-results.json')
            matches = False
            if process.returncode == 0 and result_path.exists():
                try:
                    generated = isolated_code/result_path.name
                    actual_json = (generated.read_text(encoding='utf-8')
                                   if generated.exists() else process.stdout)
                    matches = stable(json.loads(actual_json)) == stable(
                        json.loads(result_path.read_text(encoding='utf-8')))
                except json.JSONDecodeError:
                    matches = False
            checks.append({'chapter': number, 'program': program.name,
                           'exit_code': process.returncode,
                           'saved_json_matches_except_declared_volatile_fields': matches,
                           'code_sha256': hashlib.sha256(program.read_bytes()).hexdigest()})
            if process.returncode or matches is False:
                failures.append(f'{ident} program: execution/saved-output mismatch')
    finally:
        temporary.cleanup()
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'status': 'passed' if not failures else 'failed',
              'chapters': sorted(numbers), 'checks': checks, 'failures': failures,
              'volatile_fields_excluded_from_json_comparison': sorted(VOLATILE),
              'scope': 'actual isolated listing/program execution and stated '
              'output checks; independent scientific arithmetic review is separate'}
    (ROOT/'chapter-code-validation.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'chapters': sorted(numbers),
                      'checks': len(checks), 'failures': failures}, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__':
    main()
