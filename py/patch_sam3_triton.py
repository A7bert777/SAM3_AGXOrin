#!/usr/bin/env python3
"""让 sam3 在「无 Triton」的 Jetson 上可正常 import（幂等补丁）。

问题
----
``sam3/model/edt.py`` 在**模块顶层**写死了::

    import triton
    import triton.language as tl

而该模块位于 ``import sam3`` 的必经链上::

    sam3/__init__.py -> model_builder -> model.sam1_task_predictor
      -> model.sam3_tracker_base -> model.sam3_tracker_utils -> model.edt

Triton 官方无 Jetson(aarch64) wheel，于是连「文本提示图像分割」都会
``ModuleNotFoundError: No module named 'triton'``。EDT 只有视频跟踪用得到。

为什么不伪造一个假的 triton 包？
--------------------------------
因为 ``torch/utils/_triton.py::has_triton_package()`` 只做 ``import triton``。
一旦伪造成功，torch 就会误判「本机有 Triton」，后续
``torch/_inductor/runtime/hints.py`` 会继续索要 ``triton.compiler.compiler``、
``triton.backends.compiler`` 等真实实现，越陷越深。

正确做法是让 sam3 感知真实环境（无 Triton），即把 edt.py 的顶层导入
改成「可选导入 + 调用时再报错」。这样：

* ``import sam3`` 正常通过（图像分割全程可用）；
* ``has_triton_package()`` 返回 False，torch 走「无 Triton」分支，行为正确；
* 若真调用 ``edt_triton``（仅视频跟踪），会得到清晰的中文提示。

用法::

    python patch_sam3_triton.py            # 自动定位 venv 内的 sam3
    python patch_sam3_triton.py --dry-run  # 只检查，不修改
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 补丁标记：用于幂等判断
MARK = "# [SAM3_AGXOrin] triton 可选导入（Jetson 无 Triton）"

# 匹配原始的两行导入（允许中间有空行/注释的常见变体）
PATTERN = re.compile(
    r"^import triton\s*\nimport triton\.language as tl\s*$",
    re.MULTILINE,
)

REPLACEMENT = f'''{MARK}
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # Jetson 等无 Triton 的平台：图像分割不需要 EDT，放行导入
    _HAS_TRITON = False

    class _TritonStub:
        """占位对象：``@triton.jit`` 在模块级求值，必须返回可调用对象。"""

        @staticmethod
        def jit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def _deco(fn):
                return fn

            return _deco

    class _TlStub:
        """占位对象：函数注解 ``horizontal: tl.constexpr`` 在定义时会被求值。"""

        def __getattr__(self, item):
            return None

    triton = _TritonStub()
    tl = _TlStub()'''


def find_sam3_edt() -> list[Path]:
    """返回所有待处理的 edt.py 路径（venv 内 + 全局 site-packages）。"""
    found: list[Path] = []
    here = Path(__file__).resolve().parent.parent

    candidates = [
        # 本项目 venv（优先）
        here / "venv310" / "lib" / "python3.10" / "site-packages" / "sam3" / "model" / "edt.py",
        here / "venv310" / "lib64" / "python3.10" / "site-packages" / "sam3" / "model" / "edt.py",
    ]

    # 当前解释器自身的 sam3（若本来就是 venv 里的 python 运行本脚本）
    try:
        import importlib.util

        spec = importlib.util.find_spec("sam3")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).parent / "model" / "edt.py")
    except Exception:
        pass

    for path in candidates:
        if path.is_file() and path not in found:
            found.append(path)
    return found


def patch_file(path: Path, dry_run: bool = False) -> str:
    """对单个 edt.py 打补丁。返回状态描述。"""
    text = path.read_text(encoding="utf-8")

    if MARK in text:
        return f"[跳过] 已打过补丁: {path}"

    if not PATTERN.search(text):
        return f"[警告] 未找到目标导入语句，请人工检查: {path}"

    new_text, n = PATTERN.subn(REPLACEMENT, text, count=1)
    if n != 1:
        return f"[警告] 替换次数异常({n})，已跳过: {path}"

    if dry_run:
        return f"[预演] 将修改: {path}"

    # 备份一次，便于回滚
    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    path.write_text(new_text, encoding="utf-8")
    return f"[完成] 已打补丁: {path}  (备份: {backup.name})"


def main() -> int:
    parser = argparse.ArgumentParser(description="为 sam3 打「无 Triton」兼容补丁")
    parser.add_argument("--dry-run", action="store_true", help="只检查不修改")
    args = parser.parse_args()

    targets = find_sam3_edt()
    if not targets:
        print("[错误] 未找到 sam3/model/edt.py，请先执行 setup.sh 安装 sam3。")
        return 1

    rc = 0
    for path in targets:
        msg = patch_file(path, dry_run=args.dry_run)
        print(msg)
        if msg.startswith("[警告]") or msg.startswith("[错误]"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())