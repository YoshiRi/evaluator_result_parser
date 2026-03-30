#!/usr/bin/env bash
# =============================================================================
# setup_and_serve.sh — 新規 PC 向け環境構築 + t4-server 起動スクリプト (uv 版)
#
# 使い方:
#   bash setup_and_serve.sh [オプション]
#
# オプション:
#   --data-dir PATH        データセット保管ディレクトリ (デフォルト: ~/t4datasets)
#   --search-depth N       サブディレクトリ探索深さ (デフォルト: 1)
#   --port PORT            サーバーポート (デフォルト: 8000)
#   --host HOST            バインドアドレス (デフォルト: 0.0.0.0 = 全インターフェース)
#   --tier4-cache N        メモリ上に保持する Tier4 インスタンス数 (デフォルト: 8)
#   --project-id ID        webauto プロジェクト ID (WEBAUTO_PROJECT_ID 環境変数でも可)
#   --python VERSION       Python バージョン指定、例: 3.11 (デフォルト: 3.11)
#   --venv PATH            仮想環境ディレクトリ (デフォルト: .venv)
#   --skip-install         uv pip install をスキップ (再起動時など)
#   --no-server            セットアップのみ行いサーバーを起動しない
#   -h, --help             このヘルプを表示
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------
DATA_DIR="${HOME}/t4datasets"
SEARCH_DEPTH=1
HOST="0.0.0.0"
PORT=8000
TIER4_CACHE=8
WEBAUTO_PROJECT_ID="${WEBAUTO_PROJECT_ID:-}"
PYTHON_VERSION="3.11"
VENV_DIR=".venv"
SKIP_INSTALL=false
NO_SERVER=false

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
info()    { echo "[INFO]  $*"; }
success() { echo "[OK]    $*"; }
warn()    { echo "[WARN]  $*" >&2; }
error()   { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)      DATA_DIR="$2";           shift 2 ;;
        --search-depth)  SEARCH_DEPTH="$2";       shift 2 ;;
        --port)          PORT="$2";               shift 2 ;;
        --host)          HOST="$2";               shift 2 ;;
        --tier4-cache)   TIER4_CACHE="$2";        shift 2 ;;
        --project-id)    WEBAUTO_PROJECT_ID="$2"; shift 2 ;;
        --python)        PYTHON_VERSION="$2";     shift 2 ;;
        --venv)          VENV_DIR="$2";           shift 2 ;;
        --skip-install)  SKIP_INSTALL=true;       shift ;;
        --no-server)     NO_SERVER=true;          shift ;;
        -h|--help)
            sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: uv の確認
# ---------------------------------------------------------------------------
info "=== Step 1: uv 確認 ==="

if ! command -v uv &>/dev/null; then
    error "uv が見つかりません。https://docs.astral.sh/uv/getting-started/installation/ を参照してインストールしてください。"
fi
success "uv: $(uv --version)"

# ---------------------------------------------------------------------------
# Step 2: 仮想環境のセットアップ
# ---------------------------------------------------------------------------
info "=== Step 2: 仮想環境 ($VENV_DIR, Python $PYTHON_VERSION) ==="

VENV_PATH="$REPO_DIR/$VENV_DIR"

if [[ ! -d "$VENV_PATH" ]]; then
    uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
    success "仮想環境を作成しました: $VENV_PATH"
else
    success "既存の仮想環境を使用: $VENV_PATH"
fi

PYTHON_VENV="$VENV_PATH/bin/python"
success "Python: $("$PYTHON_VENV" --version)"

# ---------------------------------------------------------------------------
# Step 3: 依存ライブラリのインストール
# ---------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" == false ]]; then
    info "=== Step 3: 依存ライブラリのインストール ==="

    # t4-devkit（GitHub から、既存ならスキップ）
    if "$PYTHON_VENV" -c "import t4_devkit" 2>/dev/null; then
        success "t4_devkit は既にインストール済みです"
    else
        info "t4_devkit をインストール中..."
        uv pip install --python "$PYTHON_VENV" \
            "git+https://github.com/tier4/t4-devkit.git"
        success "t4_devkit をインストールしました"
    fi

    # パッケージ本体（pyproject.toml の依存 + server extras を一括インストール）
    info "evaluator-result-parser[server] をインストール中..."
    uv pip install --python "$PYTHON_VENV" -e "$REPO_DIR[server]"
    success "パッケージをインストールしました"
else
    info "=== Step 3: インストールをスキップ ==="
fi

# ---------------------------------------------------------------------------
# Step 4: webauto プロジェクト ID の設定確認
# ---------------------------------------------------------------------------
info "=== Step 4: webauto 設定 ==="

if [[ -n "$WEBAUTO_PROJECT_ID" ]]; then
    export WEBAUTO_PROJECT_ID
    success "WEBAUTO_PROJECT_ID = $WEBAUTO_PROJECT_ID"
else
    warn "WEBAUTO_PROJECT_ID が未設定です。"
    warn "ダウンロードを使う場合は --project-id <ID> を指定するか、"
    warn "環境変数 WEBAUTO_PROJECT_ID を設定してください。"
    warn "--no-download で既存データを使う場合は不要です。"
fi

# ---------------------------------------------------------------------------
# Step 5: データディレクトリの確認
# ---------------------------------------------------------------------------
info "=== Step 5: データディレクトリ確認 ==="

mkdir -p "$DATA_DIR"
success "データディレクトリ: $DATA_DIR"

dataset_count=0
for d in "$DATA_DIR"/*/; do
    [[ -d "$d" ]] && [[ -d "${d}annotation" || -d "${d}data" ]] && dataset_count=$((dataset_count + 1))
done
if [[ "$SEARCH_DEPTH" -ge 1 ]]; then
    for group in "$DATA_DIR"/*/; do
        for d in "$group"*/; do
            [[ -d "$d" ]] && [[ -d "${d}annotation" || -d "${d}data" ]] && dataset_count=$((dataset_count + 1))
        done
    done
fi

if [[ "$dataset_count" -gt 0 ]]; then
    success "検出済みデータセット数: ${dataset_count} (search_depth=${SEARCH_DEPTH})"
else
    warn "データセットが見つかりません (${DATA_DIR})"
    warn "サーバー起動後、webauto でダウンロードするか --data-dir を確認してください。"
fi

# ---------------------------------------------------------------------------
# Step 6: インストール確認
# ---------------------------------------------------------------------------
info "=== Step 6: インストール確認 ==="

T4SERVER="$VENV_PATH/bin/t4-server"
[[ ! -f "$T4SERVER" ]] && error "t4-server が見つかりません: $T4SERVER — Step 3 を確認してください。"
success "t4-server: $T4SERVER"

# ---------------------------------------------------------------------------
# サマリー表示
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  セットアップ完了"
echo "============================================================"
echo "  データディレクトリ : $DATA_DIR"
echo "  Search depth      : $SEARCH_DEPTH"
echo "  ホスト            : $HOST"
echo "  ポート            : $PORT"
echo "  Tier4 キャッシュ  : $TIER4_CACHE"
[[ -n "$WEBAUTO_PROJECT_ID" ]] && echo "  webauto project   : $WEBAUTO_PROJECT_ID"
echo "============================================================"
echo ""

if [[ "$NO_SERVER" == true ]]; then
    info "サーバーの起動をスキップします (--no-server)。"
    echo ""
    echo "手動起動コマンド:"
    echo "  $T4SERVER --host \"$HOST\" --port $PORT \\"
    echo "            --data-dir \"$DATA_DIR\" --search-depth $SEARCH_DEPTH \\"
    echo "            --tier4-cache $TIER4_CACHE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 7: サーバー起動
# ---------------------------------------------------------------------------
info "=== Step 7: t4-server 起動 ==="

# LAN IP を表示してネットワーク越しのアクセス先を明示する
LAN_IP=""
if command -v ip &>/dev/null; then
    LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')
elif command -v ifconfig &>/dev/null; then
    LAN_IP=$(ifconfig | awk '/inet /&&!/127.0.0.1/{print $2; exit}')
fi

echo ""
echo "  ローカル          : http://localhost:${PORT}"
[[ -n "$LAN_IP" ]] && \
echo "  ネットワーク内    : http://${LAN_IP}:${PORT}"
echo ""
echo "  API ドキュメント  : http://localhost:${PORT}/docs"
echo "  ヘルスチェック    : http://localhost:${PORT}/health"
echo "  データセット一覧  : http://localhost:${PORT}/datasets"
echo ""
echo "  停止するには Ctrl+C を押してください。"
echo "============================================================"
echo ""

exec "$T4SERVER" \
    --host "$HOST" \
    --data-dir "$DATA_DIR" \
    --search-depth "$SEARCH_DEPTH" \
    --port "$PORT" \
    --tier4-cache "$TIER4_CACHE"
