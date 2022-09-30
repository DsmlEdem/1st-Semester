import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Compute word statistics')
    parser.add_argument('path', help='The filepath containing the words')
    args = parser.parse_args()
    return args.path


def main(path):
    with open(path, encoding='utf8') as f:
        words = f.read().splitlines()

    lettergroups = defaultdict(set)
    for word in words:
        lettergroups[frozenset(word)].add(word)

    ngroups = len(lettergroups)

    max_group_len = max(map(len, lettergroups.values()))

    friendliest = []
    for group in lettergroups.values():
        if len(group) == max_group_len:
            friendliest.extend(group)
    friendliest.sort()

    print(f'{ngroups} different group(s)')
    print(f'largest group has {max_group_len} word(s)')
    print('friendliest word(s) are:')
    for x in friendliest:
        print(f'- {x}')


if __name__ == '__main__':
    main(parse_args())