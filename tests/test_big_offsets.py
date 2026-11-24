"""Weight-file positions above 4 GiB: the kernel must read them as 64-bit.   usage: python tests/test_big_offsets.py
Takes a real header, moves every offset up by 5 GiB, and checks the kernel's load_cfg reads the full 64-bit values (the old kernel returned -6)."""
import glob, os, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not shutil.which("g++"):
    for d in glob.glob(r"C:\AMDDesignTools\*\Vitis\tps\mingw\*\win64.o\nt\bin"):
        if (Path(d) / "g++.exe").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            break
if not shutil.which("g++"):
    print("SKIP  no C++ compiler found"); sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="big_"))
src = ROOT / "models" / "stories260K.flat"
if not src.exists():
    print("SKIP  models/stories260K.flat not present (convert a llama2.c checkpoint with ref/convert.py to enable this test)"); sys.exit(0)
hdr = bytearray(src.read_bytes()[:256])
h = list(struct.unpack("<64I", hdr))
SHIFT = 5 * 2 ** 30
for i in range(7):
    v = (h[12 + 2 * i] << 32 | h[11 + 2 * i]) + (SHIFT if i != 2 else 0)   # i=2 is the layer stride, not a position
    h[11 + 2 * i], h[12 + 2 * i] = v & 0xFFFFFFFF, v >> 32
want = [(h[12 + 2 * i] << 32 | h[11 + 2 * i]) // 4 for i in range(7)]
(TMP / "t.cpp").write_text('#include "llm_kernel.cpp"\n#include <stdio.h>\nint main(){ static float w[64] __attribute__((aligned(64))); FILE*f=fopen("%s","rb"); fread(w,4,64,f); fclose(f);\n'
    ' Cfg c; int rc=load_cfg((const llm_wvec_t*)w,c); printf("%%d %%llu %%llu %%llu %%llu %%llu %%llu %%llu\\n", rc,(unsigned long long)c.tok,(unsigned long long)c.lay0,(unsigned long long)c.stride,'
    '(unsigned long long)c.fin,(unsigned long long)c.cos_,(unsigned long long)c.sin_,(unsigned long long)c.wcls); }\n' % (TMP / "h.bin").as_posix())
(TMP / "h.bin").write_bytes(struct.pack("<64I", *h))
subprocess.run(["g++", "-O1", "-std=c++14", "-Wno-unknown-pragmas", f"-I{ROOT / 'sw'}", "-o", str(TMP / "t.exe"), str(TMP / "t.cpp")], check=True)
out = subprocess.run([str(TMP / "t.exe")], capture_output=True, text=True).stdout.split()
ok = out == ["0"] + [str(x) for x in want]
print(("PASS  " if ok else "FAIL  ") + f"offsets above 4 GiB are read in full (got {out})")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if ok else 1)
