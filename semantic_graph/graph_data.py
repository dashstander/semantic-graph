from collections import namedtuple

Triple = namedtuple('Triple', ['h', 'r', 't'])

def str_to_triple(triple: str):
    return Triple(*triple.split(' | '))

