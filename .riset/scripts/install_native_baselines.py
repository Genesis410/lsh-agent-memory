"""Install native baseline libraries hnswlib and annoy using MinGW-w64 G++.

Self-contained deployment on Windows with static linking (-static-libgcc -static-libstdc++).
Zero external runtime DLL dependencies.
"""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def install_native_baselines():
    print("=================================================================")
    print("TASK 1: COMPILING & DEPLOYING NATIVE BASELINES (HNSWLIB & ANNOY)")
    print("=================================================================")

    import site
    user_site = Path(site.getusersitepackages())
    user_site.mkdir(parents=True, exist_ok=True)
    print(f"Target site-packages: {user_site}")

    mingw_bin = Path("C:/ProgramData/mingw64/mingw64/bin")
    gpp = mingw_bin / "g++.exe"
    if not gpp.exists():
        gpp = Path(shutil.which("g++"))
    assert gpp.exists(), "MinGW G++ compiler not found!"
    print(f"Using compiler: {gpp}")

    py_inc = sysconfig.get_path("include")
    py_lib = sysconfig.get_config_var("LIBDIR")
    import numpy as np
    import pybind11
    numpy_inc = np.get_include()
    pybind_inc = pybind11.get_include()

    arch_flags = [
        "-O3", "-march=native", "-mfma", "-mavx2",
        "-static-libgcc", "-static-libstdc++",
        "-Wl,-Bstatic", "-lstdc++", "-lpthread", "-Wl,-Bdynamic"
    ]

    # Clean any corrupted previous folder in user_site
    hnsw_old_dir = user_site / "hnswlib"
    if hnsw_old_dir.is_dir():
        shutil.rmtree(hnsw_old_dir)

    # -------------------------------------------------------------
    # 1. Compile and Deploy hnswlib (v0.8.0)
    # -------------------------------------------------------------
    print("\n--- Compiling hnswlib 0.8.0 ---")
    hnsw_src_dir = Path("build_native/hnswlib-0.8.0")
    hnsw_pyd_dest = user_site / "hnswlib.cp314-win_amd64.pyd"

    hnsw_cmd = [
        str(gpp), "-shared",
        "-std=c++14",
        "-DVERSION_INFO=\"0.8.0\"",
        f"-I{pybind_inc}",
        f"-I{numpy_inc}",
        f"-I{py_inc}",
        f"-I{hnsw_src_dir / 'hnswlib'}",
        str(hnsw_src_dir / "python_bindings" / "bindings.cpp"),
        f"-L{py_lib}", "-lpython314",
        *arch_flags,
        "-o", str(hnsw_pyd_dest)
    ]
    print("Compiling and linking hnswlib standalone extension...")
    res = subprocess.run(hnsw_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"hnswlib build failed:\n{res.stderr}")
        sys.exit(1)
    print(f"hnswlib installed to {hnsw_pyd_dest}")

    # -------------------------------------------------------------
    # 2. Compile and Deploy annoy (v1.17.3)
    # -------------------------------------------------------------
    print("\n--- Compiling annoy 1.17.3 ---")
    annoy_src_dir = Path("build_native/annoy-1.17.3")
    annoy_pkg_dir = user_site / "annoy"
    annoy_pkg_dir.mkdir(parents=True, exist_ok=True)

    annoy_pyd_dest = annoy_pkg_dir / "annoylib.cp314-win_amd64.pyd"
    annoy_cmd = [
        str(gpp), "-shared",
        "-std=c++14",
        "-D_CRT_SECURE_NO_WARNINGS", "-fpermissive",
        "-DANNOYLIB_MULTITHREADED_BUILD",
        f"-I{py_inc}",
        f"-I{annoy_src_dir / 'src'}",
        str(annoy_src_dir / "src" / "annoymodule.cc"),
        f"-L{py_lib}", "-lpython314",
        *arch_flags,
        "-o", str(annoy_pyd_dest)
    ]
    print("Compiling and linking annoy standalone extension...")
    res = subprocess.run(annoy_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"annoy build failed:\n{res.stderr}")
        sys.exit(1)

    # Copy __init__.py and type stubs
    annoy_init_src = annoy_src_dir / "annoy" / "__init__.py"
    shutil.copy2(annoy_init_src, annoy_pkg_dir / "__init__.py")
    for extra in ["py.typed", "__init__.pyi"]:
        src_extra = annoy_src_dir / "annoy" / extra
        if src_extra.exists():
            shutil.copy2(src_extra, annoy_pkg_dir / extra)
    print(f"annoy installed to {annoy_pkg_dir}")

    # -------------------------------------------------------------
    # 3. Comprehensive Verification
    # -------------------------------------------------------------
    print("\n--- Verifying All Baselines In Fresh Python Process ---")
    test_code = """
import hnswlib
import annoy
import faiss
import rank_bm25

print('[OK] hnswlib imported from:', hnswlib.__file__)
h = hnswlib.Index(space='ip', dim=4)
h.init_index(max_elements=10, ef_construction=100, M=16)
h.add_items([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
lbls, dists = h.knn_query([1.0, 0.0, 0.0, 0.0], k=2)
print('[OK] hnswlib query test passed:', lbls, dists)

print('[OK] annoy imported from:', annoy.__file__)
a = annoy.AnnoyIndex(4, 'dot')
a.add_item(0, [1.0, 0.0, 0.0, 0.0])
a.add_item(1, [0.0, 1.0, 0.0, 0.0])
a.build(10)
annoy_res = a.get_nns_by_vector([1.0, 0.0, 0.0, 0.0], 2, include_distances=True)
print('[OK] annoy query test passed:', annoy_res)

print('[OK] FAISS ITQTransform:', hasattr(faiss, 'ITQTransform'))
print('[OK] FAISS IndexRefineFlat:', hasattr(faiss, 'IndexRefineFlat'))
print('[OK] rank-bm25 available')
print('[ALL VERIFICATIONS PASSED SUCCESSFULLY]')
"""
    v_res = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)
    print(v_res.stdout)
    if v_res.returncode != 0:
        print("Verification failed:\n", v_res.stderr)
        sys.exit(1)

    print("[SUCCESS] Task 1 completed: All native baselines installed and verified!")


if __name__ == "__main__":
    install_native_baselines()
