"""Check reader integrity and behavior; scientific acceptance is a separate gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BOOK-CONTENT-START -->'
END = '<!-- BOOK-CONTENT-END -->'


def static_checks():
    failures, pages = [], []
    course = json.loads((ROOT/'course.json').read_text(encoding='utf-8'))
    durations = {c['video_id']:c['duration'] for c in course['chapters']}
    for path in sorted(ROOT.glob('*.html')):
        if path.name in ('book-overview.html','laboratory-body.html'):
            continue
        raw = path.read_bytes().decode('utf-8')
        soup = BeautifulSoup(raw,'html.parser')
        ids = [e['id'] for e in soup.select('[id]')]
        if len(ids) != len(set(ids)):
            failures.append(f'{path.name}: duplicate HTML IDs')
        if len(soup.select('h1')) != 1:
            failures.append(f'{path.name}: expected exactly one h1')
        links = soup.select('a[href]')+soup.select('script[src]')+soup.select('link[href]')+soup.select('img[src]')
        for element in links:
            value = element.get('href',element.get('src'))
            url = urllib.parse.urlsplit(value)
            if url.scheme or url.netloc:
                if url.netloc in ('www.youtube.com','youtube.com'):
                    query = urllib.parse.parse_qs(url.query)
                    identifier = query.get('v',[None])[0]
                    if identifier in durations and 't' in query:
                        seconds = int(query['t'][0].rstrip('s'))
                        if not 0 <= seconds <= durations[identifier]:
                            failures.append(f'{path.name}: out-of-bounds timestamp {value}')
                continue
            target = (path.parent/urllib.parse.unquote(url.path)).resolve() if url.path else path
            if not target.is_file():
                failures.append(f'{path.name}: missing local file {value}')
                continue
            if url.fragment and target.suffix == '.html':
                target_soup = soup if target == path else BeautifulSoup(target.read_text(encoding='utf-8'),'html.parser')
                if target_soup.find(id=urllib.parse.unquote(url.fragment)) is None:
                    failures.append(f'{path.name}: missing local anchor {value}')
        chapter_match = re.fullmatch(r'chapter-(\d\d)\.html',path.name)
        body_digest = None
        if chapter_match:
            source = ROOT/'chapters'/f'ch{chapter_match[1]}.html'
            data = source.read_bytes()
            body_digest = hashlib.sha256(data).hexdigest()
            retained = raw.split(START,1)[1].split(END,1)[0].encode('utf-8')
            if retained != data:
                failures.append(f'{path.name}: wrapped authored body differs')
            if soup.find('meta',attrs={'name':'authored-body-sha256'})['content'] != body_digest:
                failures.append(f'{path.name}: digest marker differs')
            chapter = next(c for c in course['chapters'] if c['id'] == 'ch'+chapter_match[1])
            if chapter['status'] == 'accepted and locally validated' and chapter.get('accepted_body_sha256') != body_digest:
                failures.append(f'{path.name}: acceptance is stale')
            meta_path=source.with_suffix('.meta.json')
            if not meta_path.exists():
                failures.append(f'{path.name}: missing chapter provenance metadata')
            else:
                metadata=json.loads(meta_path.read_text(encoding='utf-8'))
                for section in metadata.get('sections',[]):
                    anchor=section.get('id')
                    if not anchor or anchor not in ids:
                        failures.append(f'{path.name}: provenance section has no matching anchor {anchor}')
                    start,end=section.get('start'),section.get('end')
                    if start is not None or end is not None:
                        duration=durations.get(section.get('video_id'))
                        if (not isinstance(start,(int,float)) or not isinstance(end,(int,float))
                                or duration is None or not 0 <= start <= end <= duration):
                            failures.append(f'{path.name}: invalid provenance recording span {anchor}')
        main = soup.select_one('#chapter-content')
        if main:
            for e in main.select('pre,script,style'):
                e.decompose()
            text = main.get_text(' ',strip=True)
        else:
            text = ''
        for pattern in [r'[A-Z]:\\Users\\',r'sk-proj-[A-Za-z0-9_-]+',r'(?i)(?:api_key|access_token)\s*=\s*[\'"][^\'"]{12,}']:
            if re.search(pattern,raw):
                failures.append(f'{path.name}: privacy pattern {pattern}')
        pages.append({'file':path.name,'prose_diagnostic_words':len(text.split()),
                      'h2':len(soup.select('article h2')),
                      'solutions':len(soup.select('details.solution')),
                      'body_sha256':body_digest})
    return {'status':'passed' if not failures else 'failed','failures':failures,'pages':pages}


def browser_checks():
    from playwright.sync_api import sync_playwright
    results, failures, requests = [], [], []
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel='chrome',headless=True)
        except Exception:
            browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        context.add_init_script("""Object.defineProperty(navigator, 'clipboard', {
            value: {writeText: async text => {window.__copiedText = text;}}
        });""")
        def route_handler(route):
            if route.request.url.startswith(('http:','https:')):
                requests.append(route.request.url)
                route.abort()
            else:
                route.continue_()
        context.route('**/*',route_handler)
        page = context.new_page()
        errors = []
        page.on('pageerror',lambda error:errors.append(str(error)))
        files = [p for p in sorted(ROOT.glob('*.html'))
                 if p.name not in ('book-overview.html','laboratory-body.html')]
        for path in files:
            for width in (1280,390):
                page.set_viewport_size({'width':width,'height':900})
                errors.clear()
                page.goto(path.as_uri(),wait_until='load')
                page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.promise')
                page.evaluate('() => MathJax.startup.promise')
                page.evaluate('''async () => {
                    await Promise.all([...document.images].map(async img => {
                        img.loading = 'eager';
                        try { await img.decode(); } catch {}
                    }));
                }''')
                metrics = page.evaluate('''() => ({
                    mathErrors:[...document.querySelectorAll('mjx-merror,[data-mml-node="merror"]')].map(e=>e.textContent),
                    renderedMath:document.querySelectorAll('mjx-container').length,
                    overflow:document.documentElement.scrollWidth>innerWidth+2,
                    h1:document.querySelector('h1')?.textContent,
                    brokenImages:[...document.images].filter(e=>!e.complete||!e.naturalWidth).length
                })''')
                for key in ('mathErrors','overflow','brokenImages'):
                    if metrics[key]: failures.append(f'{path.name}@{width}: {key}={metrics[key]}')
                if errors: failures.append(f'{path.name}@{width}: JavaScript errors {errors}')
                page.locator('#theme-toggle').click()
                first = page.evaluate('document.documentElement.classList.contains("light")')
                page.locator('#theme-toggle').click()
                if page.evaluate('document.documentElement.classList.contains("light")') == first:
                    failures.append(f'{path.name}@{width}: theme does not toggle')
                page.locator('#text-toggle').click()
                if not page.evaluate('document.documentElement.classList.contains("large-text")'):
                    failures.append(f'{path.name}@{width}: larger text does not toggle')
                if page.evaluate('document.documentElement.scrollWidth>innerWidth+2'):
                    failures.append(f'{path.name}@{width}: overflow with larger text')
                page.locator('#text-toggle').click()
                solutions = page.locator('details.solution')
                if solutions.count():
                    page.locator('#solutions-toggle').click()
                    if page.locator('details.solution:not([open])').count():
                        failures.append(f'{path.name}@{width}: solutions do not expand')
                    page.locator('#text-toggle').click()
                    if page.evaluate('document.documentElement.scrollWidth>innerWidth+2'):
                        failures.append(f'{path.name}@{width}: overflow with solutions and larger text')
                    page.locator('#text-toggle').click()
                    page.locator('#solutions-toggle').click()
                if page.locator('.copybtn').count():
                    expected = page.locator('pre').first.text_content()
                    page.locator('.copybtn').first.click()
                    page.wait_for_function('window.__copiedText !== undefined')
                    if page.evaluate('window.__copiedText') != expected:
                        failures.append(f'{path.name}@{width}: copy handler text mismatch')
                states = page.evaluate('[...document.querySelectorAll("details")].map(e=>e.open)')
                page.evaluate('dispatchEvent(new Event("beforeprint"))')
                if page.locator('details:not([open])').count():
                    failures.append(f'{path.name}@{width}: print does not open solutions')
                page.evaluate('dispatchEvent(new Event("afterprint"))')
                if states != page.evaluate('[...document.querySelectorAll("details")].map(e=>e.open)'):
                    failures.append(f'{path.name}@{width}: print changes solution state')
                results.append({'file':path.name,'width':width,**metrics})
        page.goto((ROOT/'index.html').as_uri(),wait_until='load')
        page.locator('#lecture-search').fill('score')
        if page.locator('[data-chapter]:visible').count()!=1:
            failures.append('Chapter search: expected one score chapter')
        context.close(); browser.close()
    if requests: failures.append('Offline reader attempted external requests: '+str(requests))
    return {'status':'passed' if not failures else 'failed','failures':failures,
            'checks':results,'external_requests_blocked':requests,
            'clipboard':'copy handler tested against isolated clipboard API stub; real system clipboard not verified',
            'actual_print_dialog':'not invoked; before/after hooks tested',
            'accessibility':'automated controls/layout checks only; no human certification'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser',action='store_true')
    args=parser.parse_args()
    static = static_checks()
    rendered = browser_checks() if args.browser else {'status':'not performed'}
    report = {'checked_at':datetime.now(timezone.utc).isoformat(),'static':static,'browser':rendered,
              'scientific_acceptance':'separate per-chapter review; not established by this program'}
    (ROOT/'reader-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'static':static['status'],'browser':rendered['status'],
                      'failures':static['failures']+rendered.get('failures',[])},indent=2))
    raise SystemExit(1 if static['status']=='failed' or rendered['status']=='failed' else 0)


if __name__ == '__main__':
    main()
