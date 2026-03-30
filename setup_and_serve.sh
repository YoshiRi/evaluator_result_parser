#!/usr/bin/env bash
# =============================================================================
# setup_and_serve.sh — 新規 PC 向け環境構築 + t4-server 起動スクリプト
#
# 使い方:
#   bash setup_and_serve.sh [オプション]
#
# オプション:
#   --data-dir PATH        データセット保管ディレクトリ (デフォルト: ~/t4datasets)
#   --search-depth N       サブディレクトリ探索深さ (デフォルト: 1)
#   --port PORT            サーバーポート (デフォルト: 8000)
#   --tier4-cache N        メモリ上に保持する Tier4 インスタンス数 (デフォルト: 8)
#   --project-id ID        webauto プロジェクト ID (WEBAUTO_PROJECT_ID 環境変数でも可)
#   --venv PATH            仮想環境ディレクトリ (デフォルト: .venv)
#   --skip-install         pip install をスキップ (再起動時など)
#   --no-server            セットアップのみ行いサーバーを起動しない
#   -h, --help             このヘルプを表示
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------
DATA_DIR="${HOME}/t4datasets"
SEARCH_DEPTH=1
PORT=8000
TIER4_CACHE=8
WEBAUTO_PROJECT_ID="${WEBAUTO_PROJECT_ID:-}"
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
        --data-dir)      DATA_DIR="$2";          shift 2 ;;
        --search-depth)  SEARCH_DEPTH="$2";      shift 2 ;;
        --port)          PORT="$2";              shift 2 ;;
        --tier4-cache)   TIER4_CACHE="$2";       shift 2 ;;
        --project-id)    WEBAUTO_PROJECT_ID="$2"; shift 2 ;;
        --venv)          VENV_DIR="$2";          shift 2 ;;
        --skip-install)  SKIP_INSTALL=true;      shift ;;
        --no-server)     NO_SERVER=true;         shift ;;
        -h|--help)
            sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Python バージョン確認
# ---------------------------------------------------------------------------
info "=== Step 1: Python 確認 ==="

PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" -ge 3 && "$minor" -ge 9 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

[[ -z "$PYTHON" ]] && error "Python 3.9 以上が見つかりません。インストールしてください。"
success "Python: $($PYTHON --version)"

# ---------------------------------------------------------------------------
# Step 2: 仮想環境のセットアップ
# ---------------------------------------------------------------------------
info "=== Step 2: 仮想環境 ($VENV_DIR) ==="

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
    success "仮想環境を作成しました: $VENV_DIR"
else
    success "既存の仮想環境を使用: $VENV_DIR"
fi

# 仮想環境内の pip / python を使う
PIP="$REPO_DIR/$VENV_DIR/bin/pip"
PYTHON_VENV="$REPO_DIR/$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# Step 3: 依存ライブラリのインストール
# ---------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" == false ]]; then
    info "=== Step 3: 依存ライブラリのインストール ==="

    "$PIP" install --upgrade pip --quiet

    # pyarrow は pandas の parquet サポートに必要
    "$PIP" install pandas numpy matplotlib Pillow pyarrow seaborn --quiet
    success "基本ライブラリをインストールしました"

    # t4-devkit（GitHub から）
    if "$PYTHON_VENV" -c "import t4_devkit" 2>/dev/null; then
        success "t4_devkit は既にインストール済みです"
    else
        info "t4_devkit をインストール中..."
        "$PIP" install "git+https://github.com/tier4/t4-devkit.git" --quiet
        success "t4_devkit をインストールしました"
    fi

    # パッケージ本体（サーバー extras 込み）
    info "evaluator-result-parser[server] をインストール中..."
    "$PIP" install -e "$REPO_DIR[server]" --quiet
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

# データセット数を確認
dataset_count=0
if [[ -d "$DATA_DIR" ]]; then
    # フラット (depth=0)
    for d in "$DATA_DIR"/*/; do
        [[ -d "$d" ]] && [[ -d "${d}annotation" || -d "${d}data" ]] && dataset_count=$((dataset_count + 1))
    done
    # 1段ネスト (depth=1)
    if [[ "$SEARCH_DEPTH" -ge 1 ]]; then
        for group in "$DATA_DIR"/*/; do
            for d in "$group"*/; do
                [[ -d "$d" ]] && [[ -d "${d}annotation" || -d "${d}data" ]] && dataset_count=$((dataset_count + 1))
            done
        done
    fi
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

T4SERVER="$REPO_DIR/$VENV_DIR/bin/t4-server"
if [[ ! -f "$T4SERVER" ]]; then
    T4SERVER="$($PYTHON_VENV -c 'import shutil; print(shutil.which("t4-server") or "")')"
fi
[[ -z "$T4SERVER" ]] && error "t4-server コマンドが見つかりません。Step 3 を確認してください。"
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
echo "  ポート            : $PORT"
echo "  Tier4 キャッシュ  : $TIER4_CACHE"
[[ -n "$WEBAUTO_PROJECT_ID" ]] && echo "  webauto project   : $WEBAUTO_PROJECT_ID"
echo "============================================================"
echo ""

if [[ "$NO_SERVER" == true ]]; then
    info "サーバーの起動をスキップします (--no-server)。"
    echo ""
    echo "手動起動コマンド:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  t4-server --data-dir \"$DATA_DIR\" --search-depth $SEARCH_DEPTH \\"
    echo "            --port $PORT --tier4-cache $TIER4_CACHE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 7: サーバー起動
# ---------------------------------------------------------------------------
info "=== Step 7: t4-server 起動 ==="
echo ""
echo "  API ドキュメント: http://localhost:${PORT}/docs"
echo "  ヘルスチェック  : http://localhost:${PORT}/health"
echo "  データセット一覧: http://localhost:${PORT}/datasets"
echo ""
echo "  停止するには Ctrl+C を押してください。"
echo "============================================================"
echo ""

exec "$T4SERVER" \
    --data-dir "$DATA_DIR" \
    --search-depth "$SEARCH_DEPTH" \
    --port "$PORT" \
    --tier4-cache "$TIER4_CACHE"
