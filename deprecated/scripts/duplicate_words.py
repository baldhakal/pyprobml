#!/usr/bin/env python3

                        # Author: Iain Murray
# Used to find a common typo in my notes: a repeated word.
# Vim spell checker sometimes highlights, but not always.

import superimport

import fileinput
import sys
import re

filenames = sys.argv[1:]

def scan_file(fid):
    dups = []
    last_word = None
    for line in fid:
        if not line.strip():
            # prevent common false positive with heading and then opening of
            # first sentence being same word
            last_word = None
        for word in line.strip().split():
            if word == last_word:
                dups.append(word)
            last_word = word if re.match(r'.*[a-zA-Z]', word) else None
    return dups

if len(sys.argv) > 1:
    for filename in sys.argv[1:]:
        with open(filename, 'r') as fid:
            if dups := scan_file(fid):
                print(f'Duplicate words seen in {filename}')
                print('\n'.join(dups))
                print('')
else:
    dups = scan_file(sys.stdin)
    print('\n'.join(dups))
