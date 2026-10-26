"""Wrap immutable authored HTML; build a reader without editing chapter bodies."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def esc(value):
    return html.escape(str(value), quote=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shell(title, body, course, chapter=None, digest=None):
    nav = '<a href="index.html">Chapters</a><a href="source-map.html">Source map</a><a href="references.html">References</a><a href="laboratory.html">Laboratory</a>'
    toc = '<aside id="chapter-toc" class="toc" aria-label="On this page"><h2>On this page</h2></aside>' if chapter else ''
    turn = ''
    if chapter:
        ids = course['_available_ids']
        position = ids.index(chapter['id'])
        def link(ident, label):
            return f'<a href="chapter-{int(ident[2:]):02d}.html">{label}</a>'
        before = link(ids[position-1], 'Previous available chapter') if position else '<span>First chapter</span>'
        after = link(ids[position+1], 'Next available chapter') if position+1 < len(ids) else '<span>End of available chapters</span>'
        turn = f'<nav class="page-turn" aria-label="Chapter navigation">{before}<a href="index.html">Chapter index</a>{after}</nav>'
    marker = f'<meta name="authored-body-sha256" content="{digest}">' if digest else ''
    guarded = '<!-- BOOK-CONTENT-START -->'+body+'<!-- BOOK-CONTENT-END -->'
    status = ('<p class="source-note">Chapter status: '+esc(chapter['status'])+'.</p>') if chapter else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">{marker}<title>{esc(title)} · MIT 6.S184 study edition</title>
<link rel="stylesheet" href="assets/book.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true}},svg:{{fontCache:'local'}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script defer src="assets/mathjax-tex-svg.js"></script><script defer src="assets/book.js"></script></head>
<body><a class="skip" href="#main">Skip to content</a><header class="site-header"><a class="brand" href="index.html">MIT 6.S184 · Flow Matching and Diffusion Models · 2026</a><nav class="course-nav" aria-label="Book">{nav}</nav></header>
<div class="layout{'' if chapter else ' no-toc'}">{toc}<main id="main"><div class="reading-tools" aria-label="Reading controls"><button id="theme-toggle" aria-pressed="false">Light theme</button><button id="text-toggle" aria-pressed="false">Larger text</button><button id="solutions-toggle" aria-pressed="false">Expand solutions</button><button id="print-page">Print with solutions</button></div>
{status}<article id="chapter-content">{guarded}</article>{turn}</main></div>
<footer class="site-footer">Independent study edition. MIT's 2026 course materials supply the sequence; extensions and teaching constructions have separate attribution. Editorial and validation status are recorded per chapter. Original sources retain their own rights.</footer></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-drafts', action='store_true')
    args = parser.parse_args()
    course = json.loads((ROOT/'course.json').read_text(encoding='utf-8'))
    course['_available_ids'] = [c['id'] for c in course['chapters']
                                if (ROOT/'chapters'/(c['id']+'.html')).exists()
                                and (c['status'] != 'pending' or args.include_drafts)]
    built, refs, spans, cards = [], {}, [], []
    for chapter in course['chapters']:
        source = ROOT/'chapters'/(chapter['id']+'.html')
        target = f'chapter-{int(chapter["id"][2:]):02d}.html'
        available = source.exists() and (chapter['status'] != 'pending' or args.include_drafts)
        accepted = chapter['status'] == 'accepted and locally validated'
        title = esc(chapter['title'])
        headline = f'<a href="{target}">{title}</a>' if available else title
        status = chapter['status'] if available else 'Planned chapter'
        cards.append(f'<li data-chapter><p class="number">CHAPTER {int(chapter["id"][2:]):02d} · LECTURE {esc(chapter["lecture"])}</p><h2>{headline}</h2><span class="status{" accepted" if accepted else ""}">{esc(status)}</span><p class="lecture-meta"><a href="https://www.youtube.com/watch?v={chapter["video_id"]}">Course recording</a> · {chapter["duration"]//60}:{chapter["duration"]%60:02d}</p></li>')
        if not available:
            continue
        data = source.read_bytes()
        body = data.decode('utf-8')
        digest = sha(data)
        if accepted and chapter.get('accepted_body_sha256') != digest:
            raise ValueError(f'{chapter["id"]}: body changed after acceptance')
        result = shell(chapter['title'], body, course, chapter, digest)
        retained = result.split('<!-- BOOK-CONTENT-START -->',1)[1].split('<!-- BOOK-CONTENT-END -->',1)[0]
        if retained.encode('utf-8') != data:
            raise AssertionError('Chapter body changed in wrapping')
        (ROOT/target).write_bytes(result.encode('utf-8'))
        built.append({'id':chapter['id'], 'file':target, 'body_sha256':digest, 'status':chapter['status']})
        meta_path = source.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        for ref in meta.get('references',[]):
            refs[(ref['url'],ref.get('locator',''))] = ref
        spans.extend(dict(s,chapter=target) for s in meta.get('sections',[]))
    intro = (ROOT/'book-overview.html').read_text(encoding='utf-8')
    accepted_count=sum(c['status']=='accepted and locally validated' for c in course['chapters'])
    progress=f'{len(built)} manuscripts available; {accepted_count} of {len(course["chapters"])} chapters independently reviewed and locally validated.'
    index = '<p class="eyebrow">Independent mathematical study edition</p><h1>Flow Matching and Diffusion Models</h1><p class="intro">From probability transport to stochastic sampling, conditional generation, latent models, and discrete diffusion.</p><p>Based on <a href="'+esc(course['source_url'])+'">MIT 6.S184, 2026</a>. Five numbered lectures form six chapters because Lecture 3 has two recordings. '+progress+' Status badges distinguish drafts from accepted chapters.</p>'+intro+'<h2 id="chapters">Chapters</h2><label class="search-label" for="lecture-search">Search chapter titles and status</label><input type="search" id="lecture-search" placeholder="For example: score, latent, reviewed"><p id="search-status" class="source-note" role="status" aria-live="polite">6 chapters shown</p><ol class="chapter-list">'+''.join(cards)+'</ol>'
    (ROOT/'index.html').write_bytes(shell('Reading guide',index,course).encode('utf-8'))
    rows = []
    for span in spans:
        anchor = span.get('id', '')
        name = span.get('title', anchor)
        start, end = span.get('start'), span.get('end')
        fields = [f'<a href="{esc(span["chapter"])}#{esc(anchor)}">{esc(name)}</a>',esc(f'{start}–{end} s'),esc(span.get('lecture_material','')),esc(span.get('research_extension','')),esc(span.get('teaching_construction',''))]
        rows.append('<tr>'+''.join('<td>'+f+'</td>' for f in fields)+'</tr>')
    source = '<h1>Sources and coverage</h1><p>The exact 2026 course edition is the organizing spine. Recording links locate topics; they do not attribute every derivation or example to the instructors. Raw caption tracks, slides, course PDFs, and downloaded papers remain outside this book.</p><p><a href="source-manifest.json">Source manifest</a> · <a href="source-map.json">Section provenance</a></p><p>Automatic captions establish approximate sequence and timing, with mathematics checked against official written sources. Caption endpoint reach and interval coverage are different diagnostics. Full audiovisual review is not implied.</p><table><thead><tr><th>Chapter section</th><th>Recording span</th><th>Lecture material</th><th>Research extension</th><th>Original construction</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    source += '''<h2 id="source-notes">Reading the source conventions</h2>
<p>The 2026 course URL, IAP label, and linked filenames identify the selected edition. The slide covers also print January 2025 dates; that inconsistency is retained as a source limitation rather than treated as verification of exact delivery dates.</p>
<p>The recording order takes precedence when it differs from the written notes. In particular, score matching precedes stochastic sampling in Lecture 3-A, and latent representations precede architectures in Lecture 4. Chapter authors read their complete assigned caption text and checked technical claims against written sources; this does not establish full visual or audiovisual review.</p>
<p>Noise schedules, finite-risk hypotheses, and endpoint limits are stated beside the relevant results. The book derives Gaussian log scores and distinguishes them from density gradients. The discrete chapter explicitly translates the source's column-generator convention into its declared row convention. These qualifications are part of the mathematical exposition, rather than assumptions inferred from abbreviated source statements.</p>'''
    (ROOT/'source-map.html').write_bytes(shell('Source map',source,course).encode('utf-8'))
    (ROOT/'source-map.json').write_text(json.dumps(spans,indent=2)+'\n',encoding='utf-8')
    bibliography = '<h1>Primary references</h1><p>Each entry records the claim and locator used by an available chapter. A source is not evidence for claims outside that scope.</p><ol>'
    for reference in refs.values():
        bibliography += '<li><a href="'+esc(reference['url'])+'">'+esc(reference['title'])+'</a>. '+esc(reference.get('locator',''))+'<p>'+esc(reference.get('supports',''))+'</p></li>'
    bibliography += '</ol>'
    (ROOT/'references.html').write_bytes(shell('Primary references',bibliography,course).encode('utf-8'))
    lab = (ROOT/'laboratory-body.html').read_text(encoding='utf-8')
    program_rows=[]
    for chapter in course['chapters']:
        if chapter['id'] not in course['_available_ids']:
            continue
        ident=chapter['id']
        number=int(ident[2:])
        program=f'code/{ident}.py'
        results=f'code/{ident}-results.json'
        if (ROOT/program).exists():
            result_link=(f' · <a href="{results}">Saved results</a>'
                         if (ROOT/results).exists() else '')
            program_rows.append(f'<li><a href="chapter-{number:02d}.html">Chapter {number}: {esc(chapter["title"])}</a> · <a href="{program}">Python program</a>{result_link}</li>')
    lab=lab.replace('<!-- CHAPTER-PROGRAMS -->','<ul>'+''.join(program_rows)+'</ul>')
    (ROOT/'laboratory.html').write_bytes(shell('Numerical laboratory',lab,course).encode('utf-8'))
    (ROOT/'build-record.json').write_text(json.dumps({'chapters':built,'authored_bodies_preserved':True},indent=2)+'\n',encoding='utf-8')
    print(f'Built {len(built)} chapter pages plus index, source map, references, laboratory.')


if __name__ == '__main__':
    main()
