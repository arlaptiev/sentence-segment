import sys
import json
import collections
import segmenter as seg


def read_in():
    '''Read data from stdin'''
    lines = sys.stdin.readlines()
    # Since an input would only be having one line, parse JSON data from that
    return json.loads(lines[0])


def flatten(lst):
    '''Flattens a list into 1-dimensional'''
    for el in lst:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_segments(sent):
    '''Recusrively predicts multiple (more than just two) segments of a sentence'''
    segments = seg.Segmenter.segment_sent(sent)

    if segments[0] == sent:
        return [sent]

    return [get_segments(segments[0]), get_segments(segments[1])]


def main():
    '''Get data as a string from read_in() and write out the segments'''
    line = read_in()

    segments = get_segments(line)

    for s in flatten(segments):
        print("%s%s" % (s[0].upper(), s[1:]) + '.')


# start process
if __name__ == '__main__':
    main()
