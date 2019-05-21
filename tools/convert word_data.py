# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:58:20 2019

@author: Kiooola
"""

original = "word_data.pkl"
destination = "word_data_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))