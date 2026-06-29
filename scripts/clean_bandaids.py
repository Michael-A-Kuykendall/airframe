#!/usr/bin/env python3
"""Remove all #![allow] bandaids added during CI debugging."""
import os, glob, re

files = glob.glob('C:/Users/micha/repos/airframe/src/bin/*.rs')
files.append('C:/Users/micha/repos/airframe/src/lib.rs')

# Pattern: any line(s) that are clearly our added bandaids
PATTERNS = [
    re.compile(r'// Cross-platform.*\n#!\[allow\(.*\)\]\n\n', re.MULTILINE),
    re.compile(r'// Cross-platform clippy.*\n(#!\[allow\([^\]]+\)\]\n)+', re.MULTILINE),
]

removed = 0
for path in files:
    content = open(path, encoding='utf-8').read()
    original = content
    for pat in PATTERNS:
        content = pat.sub('', content)
    if content != original:
        open(path, 'w', encoding='utf-8').write(content)
        print(f'cleaned: {os.path.basename(path)}')
        removed += 1

print(f'Removed bandaids from {removed} files')
