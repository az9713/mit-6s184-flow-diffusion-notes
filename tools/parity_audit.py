"""Comparable body-only diagnostics; counts flag omissions, not teaching quality."""
import argparse
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup


def measure(path):
    raw=path.read_text(encoding='utf-8')
    if path.suffix=='.html':
        soup=BeautifulSoup(raw,'html.parser')
        main=soup.select_one('#chapter-content') or soup
        headings=len(main.select('h2'))
        codes=sum('python' in ' '.join(e.get('class', []))
                  or bool(re.search(r'(^|\n)\s*(import |from |def |print\()', e.get_text()))
                  for e in main.select('pre code'))
        refs=len(set(e['href'] for e in main.select('a[href]') if e['href'].startswith('http')))
        solutions=len(main.select('details.solution'))
        for e in main.select('pre,script,style,nav'):
            e.decompose()
        text=main.get_text(' ',strip=True)
    else:
        headings=len(re.findall(r'^## ',raw,re.M))
        codes=len(re.findall(r'^```python',raw,re.M))
        refs=len(set(re.findall(r'\]\((https?://[^)]+)\)',raw)))
        solutions=None
        text=re.sub(r'```[\s\S]*?```',' ',raw)
        text=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',text)
    display=len(re.findall(r'\$\$[\s\S]*?\$\$',text))
    text=re.sub(r'\$\$[\s\S]*?\$\$',' ',text)
    text=re.sub(r'(?<!\\)\$[^$]*?(?<!\\)\$',' ',text)
    return {'file':path.name,'prose_words_excluding_math_and_code':len(text.split()),
            'major_sections':headings,'python_blocks':codes,
            'unique_external_links_including_recordings':refs,
            'display_equations':display,'expandable_solutions':solutions}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('paths',type=Path,nargs='+')
    args=parser.parse_args()
    print(json.dumps([measure(p) for p in args.paths],indent=2))
