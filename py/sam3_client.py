#!/usr/bin/env python3
"""SAM3 推理客户端：有常驻服务就用服务，没有就本地单次运行。

``sh/run.sh`` 默认调用本脚本，因此对使用者来说命令**完全不变**：

    ./sh/run.sh --image inputimage/5.jpg --point 114 285 --out outputimage/5_out_point.png

    ├─ 若 ``run/sam3.sock`` 存在且服务在线 -> 转发给常驻服务（约 1~2s 返回）
    └─ 否则                                -> 本地加载模型单次运行（约 16s，行为同以前）

这样「启动服务加速」是**可选优化**，不启动也不会出错。

环境变量：
    SAM3_SOCKET    自定义 socket 路径（需与服务端一致）
    SAM3_NO_SERVER 设为 1 时，强制本地运行（绕过服务）
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))


def socket_path() -> Path:
    env = os.environ.get("SAM3_SOCKET")
    return Path(env) if env else ROOT / "run" / "sam3.sock"


def ask_server(argv, timeout=600.0):
    """把请求发给常驻服务。

    返回 (handled, exit_code)：
      * 连接失败      -> (False, 0)，调用方应回退本地运行
      * 服务返回结果  -> (True, 0 或 1)
    """
    sp = socket_path()
    if os.environ.get("SAM3_NO_SERVER") == "1":
        return False, 0
    if not sp.exists():
        return False, 0

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(sp))
    except OSError:
        return False, 0            # 有残留文件但服务已经不在，回退本地

    try:
        with s, s.makefile("rwb") as f:
            f.write((json.dumps({"cmd": "infer", "argv": list(argv)}) + "\n").encode())
            f.flush()
            line = f.readline()
    except OSError:
        return False, 0

    if not line:
        return False, 0
    try:
        resp = json.loads(line.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return False, 0

    out = resp.get("stdout") or ""
    if out:
        sys.stdout.write(out)
    if resp.get("ok"):
        el = resp.get("elapsed_s")
        if el is not None:
            print(f"[服务] 本次请求耗时 {el}s（模型常驻，无需重新加载）")
        return True, 0
    err = resp.get("error") or "未知错误"
    print(err, file=sys.stderr)
    return True, 1


def fallback_local(argv) -> int:
    """本地单次运行（等价于直接执行 sam3_infer.py）。"""
    print("[提示] 未检测到常驻服务，本地加载模型（约 16s）。"
          "如需加速可先执行 ./sh/serve.sh start", file=sys.stderr)
    import sam3_infer
    return sam3_infer.main(list(argv))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    handled, code = ask_server(argv)
    if handled:
        return code
    return fallback_local(argv)


if __name__ == "__main__":
    sys.exit(main())