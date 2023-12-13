from math import factorial
def nr_of_pairs(n,r=2):
    """n i nr of samples, r is groupsize. """

    nCr = factorial(n) / (factorial(n - r) * factorial(r))
    return nCr


if __name__ == '__main__':
    print(nr_of_pairs(2820))
    print('done')
