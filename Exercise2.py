from collections import Counter
import heapq
import numpy as np


def huffman_encoding(input_string):
    freqs = Counter(input_string)
    heap = []
    for char, freq in freqs.items():
        heap.append([freq, [char, ""]])
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return dict(sorted(heap[0][1:], key=lambda x: len(x[1])))


def shannon_fano_encoding(input_string):
    freqs = Counter(input_string)
    symbols = [["", (char, freq)] for char, freq in freqs.items()]
    symbols.sort(key=lambda x: x[1][1], reverse=True)

    def find_optimal_pivot(symbols):
        pivot = 0
        size = len(symbols)
        diff_min = float("inf")

        for i in range(size-1):
            left_sum = sum([freq for _, (_, freq) in symbols[:i+1]])
            right_sum = sum([freq for _, (_, freq) in symbols[i+1:]])
            diff = abs(left_sum - right_sum)
            if diff < diff_min:
                diff_min = diff
                pivot = i

        return pivot

    def shannon_fano(symbols):
        if len(symbols) == 1:
            return
        elif len(symbols) == 2:
            symbols[0][0] += "0"
            symbols[1][0] += "1"
            return
        else:
            pivot = find_optimal_pivot(symbols)
            for symbol in symbols[:pivot+1]:
                symbol[0] += "0"
            for symbol in symbols[pivot+1:]:
                symbol[0] += "1"
            shannon_fano(symbols[:pivot+1])
            shannon_fano(symbols[pivot+1:])

    shannon_fano(symbols)
    shannon_fano_tree = {char: code for code, (char, _) in symbols}
    return shannon_fano_tree


def encoding_efficiency(input_string, encoding_tree):
    total_chars = len(input_string)
    total_bits = sum(len(encoding_tree[char]) for char in input_string)

    everage_code_length = total_bits / total_chars

    freqs = Counter(input_string)
    P = [freq / total_chars for freq in freqs.values()]
    H = -np.sum(P * np.log2(P))

    return H / everage_code_length


def redundancy(input_string, encoding_tree):
    return 1 - encoding_efficiency(input_string, encoding_tree)


def main():
    str = input("Enter a string: ").lower()
    huffman_tree = huffman_encoding(str)
    print("Huffman encoding:")
    for char, code in huffman_tree.items():
        print(f"\t{char}: {code}")

    encoded_str = ""
    for char in str:
        encoded_str += huffman_tree[char]

    print(f"Encoded string by Huffman encoding: {encoded_str}")
    print(
        f"Huffman encoding efficiency: {encoding_efficiency(str, huffman_tree)}")
    print(f"Huffman encoding redundancy: {redundancy(str, huffman_tree)}")

    print()
    shannon_fano_tree = shannon_fano_encoding(str)
    print("Shannon-Fano encoding:")
    for char, code in shannon_fano_tree.items():
        print(f"\t{char}: {code}")

    encoded_str = ""
    for char in str:
        encoded_str += shannon_fano_tree[char]

    print(f"Encoded string by Shannon-Fano encoding: {encoded_str}")
    print(
        f"Shannon-Fano encoding efficiency: {encoding_efficiency(str, shannon_fano_tree)}")
    print(
        f"Shannon-Fano encoding redundancy: {redundancy(str, shannon_fano_tree)}")


if __name__ == "__main__":
    main()
