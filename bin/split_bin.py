import argparse

parser = argparse.ArgumentParser(description="Split a binary file at a given byte offset.")
parser.add_argument("input",        nargs="?", default="input.bin",  help="Input file (default: input.bin)")
parser.add_argument("output1",      nargs="?", default="part1.bin",   help="First output file (default: part1.bin)")
parser.add_argument("output2",      nargs="?", default="part2.bin",   help="Second output file (default: part2.bin)")
parser.add_argument("--split", "-s", type=int,  default=1168,          help="Split point in bytes (default: 1168)")
args = parser.parse_args()

with open(args.input, "rb") as f:
    data = f.read()

with open(args.output1, "wb") as f:
    f.write(data[:args.split])

with open(args.output2, "wb") as f:
    f.write(data[args.split:])

print(f"{args.input} ({len(data)} bytes) -> {args.output1} ({args.split} bytes), {args.output2} ({len(data) - args.split} bytes)")
