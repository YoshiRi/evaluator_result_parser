# CLAUDE.md — 開発方針と設計メモ

## プロジェクト概要

自動運転評価システムの結果解析・可視化ツール集。
CSV/Parquet で指定した T4 dataset のフレームをカメラ画像＋BEV点群として可視化する。

---

## 進行中: render API 化

### 目標

`visualize_static` を「ファイル書き出し専用」から脱却させ、
**画像バイト列を返すコア関数** として整理する。
その上に CLI（既存）・HTTP サーバー（今後）を薄くかぶせる構造にする。

### 動機

- 現状の `visualize_static` は `save_dir` に PNG を書くだけで戻り値がない
- ファイルシステムを介さず bytes を返す関数があれば
  - HTTP レスポンスに直接載せられる
  - `ThreadPoolExecutor` に投げるだけで並列化できる（共有状態ゼロ）
  - テストが書きやすくなる

### レイヤー設計

```
[CLI / t4-batch / t4-multi]          [将来: HTTP server]
        |                                     |
        +------------- 共通 ----------------+
                        |
              render_frame(request) -> VisualizationResult
                        |
              visualize_static(t4, sample, ...)   ← 内部実装
                        |
                _plot_combined(...)   ← matplotlib
```

### データ型

```python
# visualize.py に追加

@dataclass
class VisualizationRequest:
    dataset_path: Path
    scenario_name: str
    frame_index: int
    target_objects: List[TargetObject] = field(default_factory=list)
    cameras: Optional[List[str]] = None
    show_annotations: bool = True
    version: Optional[str] = None
    crop_cameras: bool = False
    crop_padding: int = 40
    crop_min_size: int = 300

@dataclass
class VisualizationResult:
    camera_png: bytes        # カメラ画像（PNG）
    pointcloud_png: bytes    # BEV 点群画像（PNG）
    sample_token: str
    timestamp_us: int
```

### render_frame の実装方針

`visualize_static` 内部の `fig.savefig(path)` を
`fig.savefig(BytesIO)` に切り替えるだけで bytes が取れる。
matplotlib の `savefig` は `str | PathLike | IO` を受け付けるため変更量は最小。

```python
def render_frame(
    request: VisualizationRequest,
    t4: Optional[Tier4] = None,   # 外からキャッシュ済みを渡せる
) -> VisualizationResult:
    ...
```

`t4` 引数を省略可能にすることで:
- 省略時: 関数内で `Tier4(request.dataset_path)` をロード（単発呼び出し向け）
- 指定時: 呼び出し元がキャッシュした Tier4 を再利用（並列・サーバー向け）

### 並列化

`render_frame` は副作用なし（BytesIO はローカル変数）なので
`ThreadPoolExecutor` にそのまま投げられる。

```python
with ThreadPoolExecutor(max_workers=N) as ex:
    futures = [ex.submit(render_frame, req, t4_cache[req.dataset_path]) for req in requests]
    results = [f.result() for f in futures]
```

`Tier4` インスタンスは read-only アクセスのみなので複数スレッドからの同時参照は安全。
matplotlib は `Agg` バックエンド固定（`save_dir` があるとき既に設定済み、render_frame でも強制）。

### Tier4 キャッシュ

HTTP サーバーとして使う場合、リクエストごとに `Tier4(path)` をロードすると重い。
サーバー起動時またはオンデマンドで Tier4 インスタンスを `dict[Path, Tier4]` にキャッシュする。
このキャッシュは既存の `DatasetCache`（LRU）とは別レイヤー（Tier4オブジェクトのインメモリキャッシュ）。

---

## 実装ステップ

| Step | 内容 | ファイル | 状態 |
|------|------|---------|------|
| 1 | `VisualizationRequest` / `VisualizationResult` dataclass を追加 | `visualize.py` | done |
| 1 | `render_frame(request, t4=None) -> VisualizationResult` を実装 | `visualize.py` | done |
| 2 | `batch.py` の `visualize_frame()` を `render_frame()` ベースに差し替え | `batch.py` | done |
| 3 | HTTP サーバー実装（FastAPI 等） | `t4_visualizer/server.py` | 将来 |

---

## コーディング規約

- 新規 public 関数には docstring を書く（既存スタイルに合わせる）
- 後方互換: `visualize_static` のシグネチャは変えない（README に記載）
- matplotlib バックエンド: render 系関数では必ず `matplotlib.use("Agg")` を先頭で呼ぶ
- 型ヒント: `Optional`, `List` は `from __future__ import annotations` なしで動く形式を維持
