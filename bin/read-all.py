import sys

def main(paths):
    for path in paths:
        try:
            with open(path, 'r', encoding='utf8') as f: print(f"{path}:\n{f.read()}")
        except Exception as e: print(f"Error reading {path}: {e}")

if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths: print("Usage: python read-all.py [file1 file2 ...] ")
    else: main(paths)