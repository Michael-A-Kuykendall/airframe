#!/usr/bin/env python3
"""Scan a directory of GGUF files and print architecture, quant format, and tensor types."""
import struct, os, sys

TYPE_NAMES = {0:'F32',1:'F16',2:'Q4_0',6:'Q5_0',8:'Q8_0',12:'Q4_K',13:'Q5_K',14:'Q6_K'}
FTYPE_MAP  = {
    0:'ALL_F32',1:'F16',2:'Q4_0',7:'Q8_0',
    8:'Q3_K_S',9:'Q3_K_M',10:'Q3_K_L',
    11:'Q4_K_S',12:'Q4_K_M_old',
    14:'Q4_K_S',15:'Q4_K_M',
    16:'Q5_K_S',17:'Q5_K_M',
    18:'Q6_K',19:'Q8_0',
}
# GGUF metadata value type byte sizes (from gguf.h spec)
ESIZES = {
    0:1,  # uint8
    1:1,  # int8
    2:2,  # uint16
    3:2,  # int16
    4:4,  # uint32
    5:4,  # int32
    6:4,  # float32
    7:1,  # bool
    # 8=string, 9=array handled inline
    10:8, # uint64
    11:8, # int64
    12:8, # float64
}

def scan(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            return None, 'not_gguf'
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv      = struct.unpack('<Q', f.read(8))[0]

        arch = '?'
        ftype = '?'
        for _ in range(n_kv):
            klen = struct.unpack('<Q', f.read(8))[0]
            key  = f.read(klen).decode('utf-8', 'replace')
            vt   = struct.unpack('<I', f.read(4))[0]
            if vt == 8:  # string
                sl = struct.unpack('<Q', f.read(8))[0]
                v  = f.read(sl).decode('utf-8', 'replace')
                if 'architecture' in key:
                    arch = v
            elif vt in ESIZES:
                raw = f.read(ESIZES[vt])
                if vt == 4:  # uint32
                    v = struct.unpack('<I', raw)[0]
                    if 'file_type' in key:
                        ftype = FTYPE_MAP.get(v, str(v))
            elif vt == 9:  # array
                at = struct.unpack('<I', f.read(4))[0]
                al = struct.unpack('<Q', f.read(8))[0]
                if at == 8:
                    for _ in range(al):
                        f.read(struct.unpack('<Q', f.read(8))[0])
                elif at in ESIZES:
                    f.read(al * ESIZES[at])
                else:
                    return None, f'unknown_array_type_{at}'
            else:
                return None, f'unknown_vtype_{vt}'

        ttypes = set()
        for _ in range(n_tensors):
            nl = struct.unpack('<Q', f.read(8))[0]
            f.read(nl)
            nd = struct.unpack('<I', f.read(4))[0]
            f.read(nd * 8)
            ttypes.add(struct.unpack('<I', f.read(4))[0])
            f.read(8)

        tnames = [TYPE_NAMES.get(t, f'TYPE{t}') for t in sorted(ttypes)]
        return (arch, ftype, n_tensors, version, tnames), None


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else 'D:/shimmy-test-models/gguf_collection'
    files = sorted(f for f in os.listdir(model_dir) if f.endswith('.gguf'))
    print(f"{'Model':<55} {'Arch':<14} {'Format':<10} {'Tensor Types'}")
    print('-' * 110)
    for fname in files:
        path = os.path.join(model_dir, fname)
        mb   = os.path.getsize(path) // 1048576
        result, err = scan(path)
        if result:
            arch, ft, n, ver, tt = result
            print(f"{fname:<55} {arch:<14} {ft:<10} {tt}  ({mb} MB, v{ver})")
        else:
            print(f"{fname:<55} ERROR: {err}")


if __name__ == '__main__':
    main()
