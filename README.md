# evaluator_result_parser

自動運転評価システムの結果解析・可視化ツール集。

評価 JSONL のメトリクス分析と、T4 dataset の画像・点群可視化の 2 種類のツールを提供します。

---

## リポジトリ構成

```
evaluator_result_parser/
├── result_parser/            # 評価結果（JSONL）の解析ツール
│   ├── jsonl_parser.py       # JSONL読み込み → CSV変換
│   ├── covariance_analysis.py# 共分散 vs 誤差の散布図
│   └── metrics_visualizer.py # TPrate / mAP・距離別分析
│
└── t4_visualizer/            # T4 dataset の可視化ツール
    ├── visualize.py          # UUID + タイムスタンプで1シーンを可視化
    ├── batch.py              # CSV/Parquet から複数シーンをバッチ可視化
    ├── inspect.py            # データセット内のシーン・センサー情報を表示
    └── downloader.py         # データセットDLインターフェース（差し替え可能）
```

---

## result_parser — 評価結果解析ツール

評価システムが出力する JSONL ファイルを読み込み、検出物体のメトリクスを解析します。

### データフロー

```
JSONL files
    ↓ jsonl_parser.py
extracted_objects.csv
    ↓                    ↓
covariance_analysis.py   metrics_visualizer.py
(共分散 vs 誤差)          (TPrate / mAP / 距離別分析)
```

### jsonl_parser.py

JSONL ファイル群を読み込み、検出物体の情報を CSV に変換します。

```bash
python result_parser/jsonl_parser.py <folder_name>
# 例: python result_parser/jsonl_parser.py Nishishinjuku
# 出力: extracted_objects.csv
```

主要クラス:
- `JSONLProcessor` — フォルダ内の `.jsonl` を一括読み込み
- `ObjectExtractor` — ego 情報・検出物体（位置/速度/共分散等）を抽出
- `ObjectProcessor` — 上記を組み合わせて DataFrame を生成

### covariance_analysis.py

`extracted_objects.csv` を読み込み、推定誤差と共分散値の相関を可視化します。

```bash
python result_parser/covariance_analysis.py
```

出力: X/Y/Yaw の共分散 vs BEV/Heading 誤差の散布図（matplotlib）。

### metrics_visualizer.py

TPrate・mAP の計算と距離別分析を行います。

```bash
python result_parser/metrics_visualizer.py
```

主要機能:
- `DatasetManager.calc_tprate()` — カテゴリ別 TPrate
- `DatasetManager.calc_mAP()` — mean Average Precision
- `DatasetManager.distance_based_fp_tp_fn()` — 距離ビン別 FP/TP/FN
- `DatasetManager.distance_based_errors()` — 距離ビン別 BEV/Yaw 誤差

---

## t4_visualizer — T4 dataset 可視化ツール

[t4-devkit](https://github.com/tier4/t4-devkit) を使って T4 dataset の画像・点群を可視化します。

### インストール

```bash
pip install git+https://github.com/tier4/t4-devkit.git
pip install matplotlib pillow pandas pyarrow
```

### visualize.py — 1シーンの可視化

UUID（データセットパス）とタイムスタンプを指定して、最も近いサンプルの画像と点群を表示します。

```bash
# 基本（インタラクティブ表示）
python t4_visualizer/visualize.py /path/to/t4dataset 1609459200000000

# PNG として保存
python t4_visualizer/visualize.py /path/to/t4dataset 1609459200000000 --save-dir output/

# Rerun で3D可視化
python t4_visualizer/visualize.py /path/to/t4dataset 1609459200000000 --rerun

# 特定カメラのみ
python t4_visualizer/visualize.py /path/to/t4dataset 1609459200000000 --cameras CAM_FRONT,CAM_BACK

# バウンディングボックスなし
python t4_visualizer/visualize.py /path/to/t4dataset 1609459200000000 --no-annotations
```

出力:
- `<save_dir>/<timestamp>_cameras.png` — 全カメラ画像（2D bbox オーバーレイ付き）
- `<save_dir>/<timestamp>_pointcloud.png` — LiDAR 鳥瞰図（BEV、3D bbox フットプリント付き）

Python API としても使用できます:

```python
from t4_devkit import Tier4
from t4_visualizer.visualize import find_closest_sample, visualize_static

t4 = Tier4("/path/to/dataset")
sample = find_closest_sample(t4, timestamp_us=1609459200000000)
visualize_static(t4, sample, save_dir="output/", filename_prefix="my_scene")
```

### inspect.py — データセット情報の確認

タイムスタンプ範囲・センサーチャンネル・サンプル数を確認します。
可視化の前にまずここで確認するのが便利です。

```bash
python t4_visualizer/inspect.py /path/to/t4dataset
```

出力例:
```
============================================================
  Scenes (1)
============================================================
  name        : scene-0001
  token       : abc123...
  nbr_samples : 40

============================================================
  Samples (40)
============================================================
  Timestamp range (µs)  : 1609459200000000  →  1609459204000000
  Duration              : 4.000 s

============================================================
  Sensor Channels (from first sample)
============================================================
  CAM_FRONT                                format=jpg
  CAM_BACK                                 format=jpg
  LIDAR_TOP                                format=pcd
```

### batch.py — バッチ可視化パイプライン

CSV または Parquet で指定した複数シーンを一括で可視化します。

#### 入力ファイル形式

| カラム | 必須 | 説明 |
|---|---|---|
| `t4dataset_id` | ✅ | データセット識別子（ダウンロード・検索に使用） |
| `scenario_name` | ✅ | シーン名 |
| `frame_index` | ✅ | フレーム番号（0始まり） |
| `status` | | グループ名（例: `degrade`, `improved`）。サブフォルダに分類される |
| `cameras` | | 表示するカメラをカンマ区切りで指定（省略=全カメラ） |
| `description` | | 自由記述のメモ |
| `label` | | 対象物体のラベル文字列（例: `car`, `pedestrian`）。出力ファイル名の先頭に付与される |
| `uuid` | | 対象物体のインスタンストークン（T4 dataset の `instance_token`） |
| `x` / `y` / `z` | | 自車座標系での物体位置（メートル） |
| `width` / `length` / `height` | | 物体の BBOX サイズ（メートル） |

1 行 = 1 物体。同じ `(t4dataset_id, scenario_name, frame_index)` の行は 1 フレームにまとめて可視化されます。

入力例 (`scenes.csv`):
```csv
t4dataset_id,scenario_name,frame_index,status,label,uuid,x,y,z
dataset-abc,scene-001,0,degrade,car,inst-uuid-001,10.5,2.3,0.5
dataset-abc,scene-001,5,improved,pedestrian,inst-uuid-002,8.1,0.0,0.9
dataset-def,scene-002,10,,,,,,,
```

#### 使い方

```bash
# 永続ストレージ（デフォルト: ./t4datasets/ にDL・キャッシュ）
python t4_visualizer/batch.py scenes.csv --output-dir results/

# 一時ストレージ（終了時に自動削除、CI/ワンショット向け）
python t4_visualizer/batch.py scenes.parquet --temp --output-dir results/

# カスタムデータディレクトリ
python t4_visualizer/batch.py scenes.csv --data-dir /mnt/ssd/t4data --output-dir results/

# DL不要（既にデータがある場合）
python t4_visualizer/batch.py scenes.csv --no-download --data-dir /existing/data --output-dir results/

# 並列処理
python t4_visualizer/batch.py scenes.csv --output-dir results/ --workers 4
```

#### 出力構造

`status` 列がある場合、ステータスごとにサブフォルダへ分類されます。
各フォルダ内はフラットで、ファイル名に `<t4dataset_id>_<uuid>_<timestamp>` が付きます。

```
results/
├── batch_summary.csv                                              # 全行の成否・エラー一覧
├── degrade/
│   ├── car_dataset-abc_scene-001_f000000_cameras.png
│   ├── car_dataset-abc_scene-001_f000000_pointcloud.png
│   └── car_dataset-abc_scene-001_f000000_meta.txt
└── improved/
    ├── pedestrian_dataset-abc_scene-001_f000005_cameras.png
    └── ...
```

`label` 列がない、または空の場合は `<label>_` プレフィックスなしで出力されます。

`status` 列がない場合、全ファイルが `results/` 直下に出力されます。

`batch_summary.csv` の内容:

| カラム | 説明 |
|---|---|
| `t4dataset_id` | データセットID |
| `uuid` | シーンUUID |
| `timestamp_us` | タイムスタンプ |
| `status` | ステータス |
| `success` | 成否 |
| `output_dir` | 出力先ディレクトリ |
| `filename_prefix` | 出力ファイルのプレフィックス |
| `error` | エラーメッセージ（失敗時） |

### downloader.py — ダウンローダーの差し替え

`batch.py` が内部で呼び出すダウンローダーです。
実際のダウンロードロジックを 3 通りの方法で差し込めます。

**方法1: 既存 CLI スクリプトのパスを指定**

```python
# t4_visualizer/downloader.py を編集
DOWNLOAD_SCRIPT_PATH = "/path/to/your_download_script.py"
# → python your_download_script.py <t4dataset_id> <dest_dir> が実行される
```

**方法2: 環境変数でコマンドテンプレートを指定**

```bash
export T4_DOWNLOAD_CMD="my_tool --id {t4dataset_id} --out {dest_dir}"
python t4_visualizer/batch.py scenes.csv --output-dir results/
```

**方法3: Python 関数を直接実装**

```python
# t4_visualizer/downloader.py 内の _download_impl() を編集
def _download_impl(t4dataset_id: str, dest_dir: Path) -> None:
    import subprocess
    subprocess.run(["your_tool", t4dataset_id, str(dest_dir)], check=True)
```

ダウンロード済みのデータセットは自動的にスキップされます（キャッシュ有効）。

---

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `pandas` | CSV/Parquet の読み書き |
| `matplotlib` | 可視化（全ツール共通） |
| `seaborn` | 散布図（`covariance_analysis.py`） |
| `numpy` | 数値処理 |
| `Pillow` | 画像読み込み（`visualize.py`） |
| `pyarrow` | Parquet 読み込み（`batch.py`） |
| `t4-devkit` | T4 dataset ロード（`t4_visualizer/` 全般） |

```bash
pip install pandas matplotlib seaborn numpy pillow pyarrow
pip install git+https://github.com/tier4/t4-devkit.git
```
