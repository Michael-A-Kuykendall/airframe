import sys

def main():
    edits = []
    while len(sys.argv) > 3:
        file, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
        edits.append((file, old, new))
        del sys.argv[:4]

    for file, old, new in edits:
        try:
            with open(file, 'r', encoding='utf8') as f: content = f.read()
            if old not in content:
                print(f"Error: '{old}' not found in {file}")
                continue
            new_content = content.replace(old, new)
            with open(file, 'w', encoding='utf8') as f: f.write(new_content)
            print(f"Applied patch to {file}")
        except Exception as e:
            print(f"Error on {file}: {e}")

def main():
    if not sys.argv[1] == "patch":
        print("Usage: python batch-edit.py patch <file> <old> <new> [...]")
    else:
        main()