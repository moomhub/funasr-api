#!/usr/bin/env bash
# install_uv_service.sh
# 将基于 uv 的 Python 项目安装/卸载为 systemd 服务（不自动安装 uv）
# 日常启停请使用: systemctl start|stop|restart|status <服务名>
set -euo pipefail

# ======================== 配置区（按项目修改）========================
SERVICE_NAME="funasrapi"                 # 服务名
PROJECT_DIR="/opt/software/funasr-api"   # 项目绝对路径（**不允许包含空格**）
MAIN_SCRIPT="main.py"                    # 入口脚本
MAIN_MODULE=""                           # 或模块名，二选一
EXTRA_ARGS=""                            # 额外参数，例如 --host 0.0.0.0 --port 8000
START_MODE="uv_run"                      # uv_run 或 venv_python
RUN_USER="${SUDO_USER:-root}"            # 建议改为非 root 业务用户
UV_BIN="/usr/local/bin/uv"               # uv 可执行文件路径
SYNC_ON_INSTALL=1                        # 安装时是否执行 uv sync
USE_FROZEN_SYNC=1                        # 若有 uv.lock 则使用 --frozen
START_AFTER_INSTALL=1                    # 安装后是否立即启动

# 安全加固：生成的 systemd 服务默认启用以下保护（可改为 0 关闭）
ENABLE_SYSTEMD_HARDENING=1
# ====================================================================

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

die()  { echo "错误: $*" >&2; exit 1; }
info() { echo "==> $*"; }

need_root() {
  [[ $EUID -eq 0 ]] || die "请使用 root 或 sudo: sudo $0 $*"
}

# 检查 uv 是否可用，不可用则直接报错退出（不再自动安装）
check_uv() {
  if [[ -x "$UV_BIN" ]]; then
    return 0
  fi
  # 尝试在常见路径寻找
  for candidate in /usr/local/bin/uv /usr/bin/uv /root/.local/bin/uv "${HOME}/.local/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      UV_BIN="$candidate"
      return 0
    fi
  done
  die "未找到 uv。请先安装 uv（https://github.com/astral-sh/uv），或通过包管理器安装。"
}

# 检查运行用户是否存在
ensure_user() {
  if ! id "$RUN_USER" &>/dev/null; then
    die "系统用户 $RUN_USER 不存在。请先创建该用户或修改配置中的 RUN_USER。"
  fi
}

# 检查项目路径是否包含空格（防止 systemd 单元解析异常）
check_path_spaces() {
  if [[ "$PROJECT_DIR" =~ \  ]]; then
    die "PROJECT_DIR 包含空格，systemd 启动可能失败，请使用不含空格的路径。"
  fi
  if [[ "$MAIN_SCRIPT" =~ \  ]]; then
    die "MAIN_SCRIPT 包含空格，请重命名或修改路径。"
  fi
  if [[ "$MAIN_MODULE" =~ \  ]]; then
    die "MAIN_MODULE 包含空格，非法模块名。"
  fi
}

build_exec_start() {
  local cmd=""
  if [[ -n "$MAIN_MODULE" ]]; then
    if [[ "$START_MODE" == "venv_python" ]]; then
      cmd="${PROJECT_DIR}/.venv/bin/python -m ${MAIN_MODULE}"
    else
      cmd="${UV_BIN} run -m ${MAIN_MODULE}"
    fi
  elif [[ -n "$MAIN_SCRIPT" ]]; then
    if [[ "$START_MODE" == "venv_python" ]]; then
      cmd="${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/${MAIN_SCRIPT}"
    else
      cmd="${UV_BIN} run python ${MAIN_SCRIPT}"
    fi
  else
    die "请配置 MAIN_SCRIPT 或 MAIN_MODULE"
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    cmd="${cmd} ${EXTRA_ARGS}"
  fi
  # 清理多余空格
  echo "$cmd" | sed 's/  */ /g; s/ *$//'
}

write_service() {
  local exec_start
  exec_start="$(build_exec_start)"

  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=${SERVICE_NAME} (Python via uv)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${exec_start}
Restart=always
RestartSec=5
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}
EOF

  # 可选的安全加固
  if [[ "$ENABLE_SYSTEMD_HARDENING" == "1" ]]; then
    cat >> "$SERVICE_FILE" <<EOF
# 安全加固（可按需调整）
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=${PROJECT_DIR}
ProtectHome=yes
NoNewPrivileges=yes
RestrictRealtime=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM
EOF
  fi

  cat >> "$SERVICE_FILE" <<EOF

[Install]
WantedBy=multi-user.target
EOF

  info "已写入服务单元: $SERVICE_FILE"
  echo "    ExecStart=${exec_start}"
}

do_sync() {
  [[ "$SYNC_ON_INSTALL" == "1" ]] || return 0
  [[ -d "$PROJECT_DIR" ]] || die "项目目录不存在: $PROJECT_DIR"
  cd "$PROJECT_DIR"
  if [[ ! -f pyproject.toml && ! -f requirements.txt ]]; then
    info "未发现 pyproject.toml / requirements.txt，跳过 uv sync"
    return 0
  fi
  local -a sync_cmd=("$UV_BIN" sync)
  if [[ "$USE_FROZEN_SYNC" == "1" && -f uv.lock ]]; then
    sync_cmd+=(--frozen)
  fi
  info "同步依赖: ${sync_cmd[*]}"
  # 仅对 .venv 和必要文件调整权限，避免递归整个项目目录
  if id "$RUN_USER" &>/dev/null && [[ "$RUN_USER" != "root" ]]; then
    # 确保 .venv 存在且可写
    if [[ -d .venv ]]; then
      chown -R "${RUN_USER}:${RUN_USER}" .venv 2>/dev/null || true
    fi
    # 如果项目目录本身需要写入（如缓存文件），可适度放权
    chown "${RUN_USER}:${RUN_USER}" . 2>/dev/null || true
    sudo -u "$RUN_USER" env PATH="/usr/local/bin:/usr/bin:/bin" "${sync_cmd[@]}"
  else
    "${sync_cmd[@]}"
  fi
}

print_usage_after_install() {
  cat <<EOF

安装完成。日常管理请使用 systemd 命令：

  sudo systemctl start   ${SERVICE_NAME}
  sudo systemctl stop    ${SERVICE_NAME}
  sudo systemctl restart ${SERVICE_NAME}
  sudo systemctl status  ${SERVICE_NAME}

  # 日志查看
  sudo journalctl -u ${SERVICE_NAME} -f

开机自启：已启用（enable）
取消自启：sudo systemctl disable ${SERVICE_NAME}
重新启用：sudo systemctl enable  ${SERVICE_NAME}

卸载服务（不删除项目目录）：
  sudo $0 uninstall
EOF
}

cmd_install() {
  need_root install
  check_uv                  # 不再自动安装
  ensure_user               # 检查运行用户
  check_path_spaces         # 拒绝含空格的路径
  [[ -d "$PROJECT_DIR" ]] || die "项目目录不存在: $PROJECT_DIR"

  do_sync

  if [[ "$START_MODE" == "venv_python" && ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    die "START_MODE=venv_python 但未找到 ${PROJECT_DIR}/.venv/bin/python"
  fi

  # 调整必要文件权限（仅 .venv）
  if [[ -d "${PROJECT_DIR}/.venv" ]]; then
    chown -R "${RUN_USER}:${RUN_USER}" "${PROJECT_DIR}/.venv" 2>/dev/null || true
  fi

  write_service
  systemctl daemon-reload

  systemctl enable "$SERVICE_NAME"
  info "已设置开机自启 (systemctl enable ${SERVICE_NAME})"

  if [[ "$START_AFTER_INSTALL" == "1" ]]; then
    systemctl restart "$SERVICE_NAME"
    info "已启动服务"
    systemctl --no-pager --full status "$SERVICE_NAME" || true
  else
    info "未立即启动（START_AFTER_INSTALL=0）。需要时执行: systemctl start ${SERVICE_NAME}"
  fi

  print_usage_after_install
}

cmd_uninstall() {
  need_root uninstall

  if systemctl cat "$SERVICE_NAME" &>/dev/null; then
    info "停止服务: $SERVICE_NAME"
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    info "取消开机自启"
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
  fi

  if [[ -f "$SERVICE_FILE" ]]; then
    rm -f "$SERVICE_FILE"
    info "已删除: $SERVICE_FILE"
  else
    info "服务文件不存在: $SERVICE_FILE"
  fi

  systemctl daemon-reload
  systemctl reset-failed 2>/dev/null || true
  info "卸载完成（项目目录 ${PROJECT_DIR} 未删除）。"
}

usage() {
  cat <<EOF
用法: sudo $0 install | uninstall

  install    生成 systemd 单元、同步依赖、开机自启（可选立即启动）
  uninstall  停止服务、移除单元文件（保留项目目录）

配置摘要:
  SERVICE_NAME = ${SERVICE_NAME}
  PROJECT_DIR  = ${PROJECT_DIR}
  MAIN_SCRIPT  = ${MAIN_SCRIPT}
  MAIN_MODULE  = ${MAIN_MODULE}
  START_MODE   = ${START_MODE}
  RUN_USER     = ${RUN_USER}

安装后管理:
  sudo systemctl start|stop|restart|status ${SERVICE_NAME}
  sudo service ${SERVICE_NAME} start|stop|restart|status
  sudo journalctl -u ${SERVICE_NAME} -f
EOF
}

main() {
  case "${1:-}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    -h|--help|help|"") usage ;;
    *) die "未知命令: $1（仅支持 install / uninstall，见 --help）" ;;
  esac
}

main "$@"
