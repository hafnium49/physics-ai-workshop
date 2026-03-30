# 🤖 物理AIワークショップ

> AIに話しかけるだけで、ロボットが動き出す。プログラミング経験ゼロでOK。

---

## やること

Franka Panda ロボットアームがプレートを持ち、その上にボールが載っています。
あなたの目標：AIに日本語で指示を出しながら、ボールを落とさずに**10秒間バランスさせる**コントローラを作ること。

- 🎮 **見る** — ブラウザ上で3Dシミュレーションをリアルタイム表示。マウスで視点を自由に回転・ズーム
- 🧪 **試す** — 「この関節を動かして」「ゲインを上げて」とAIに伝えるだけ。コードは書きません
- 🏆 **競う** — コントローラスコアを改善して、リーダーボードに挑戦

---

## 💬 プログラミング経験は不要です

[Claude Code](https://claude.ai/code)（AIコーディングエージェント）があなたの言葉をコードに変換します。
ターミナルに日本語で指示を入力するだけです。例えば：

- *「ロボットを見せて」*
- *「関節6をゆっくり動かして」*
- *「なぜボールが左に落ち続けるの？」*
- *「Kpを5にして再実行して」*

---

## ⏱ ワークショップの流れ（1時間）

📍 **スプリント1（10分）探索**
ロボットの世界を見て、どの関節がプレートを傾けるか発見する

📍 **スプリント2（10分）ベースライン診断**
動かないコントローラを実行し、AIに「なぜ失敗した？」と聞く

📍 **スプリント3（10分）動く制御器**
正しいPIDコントローラを実行し、なぜ動くのかを理解する

📍 **スプリント4（15分）デジタルツイン実験**
維持マップでスコアを測定し、1つ改善して比較する

📍 **スプリント5（15分）自走型R&Dループ**
AIが仮説→編集→テスト→比較を自動で回し、最高スコアを目指す！

---

## 🚀 はじめかた

VS Codeで接続後（印刷カードを参照）：

1. ターミナルを開く：`Ctrl + ~`
2. プロンプトに `(workshop_env)` と表示されていることを確認
3. ワークスペースに移動：
   ```bash
   cd ~/physics_sim
   ```
4. Claude Codeを起動：
   ```bash
   claude
   ```
5. 最初の指示を入力：
   > 「01_validate_assembly.pyスクリプトを実行して、ライブ配信を開始して。ロボットを見たい。」
6. VS Codeの右下にポップアップが表示されたら、**「ブラウザで開く」**をクリック

---

## 🏆 コントローラスコア

コントローラの性能は**コントローラスコア**で測定されます。
ボールがプレート上のさまざまな位置からスタートし、落ちずに何秒間バランスできるかの平均値です。

| レベル | スコア | 意味 |
|--------|--------|------|
| ベースライン | ~3.3秒 | デフォルトのPID |
| 改善 | 4.0秒以上 | ゲイン調整で到達可能 |
| 優秀 | 5.0秒以上 | より賢い制御戦略が必要 |

> スコアをホストに伝えて、ホワイトボードのリーダーボードに記録しましょう！

---

## 🎮 ライブ表示の操作

ブラウザ上のシミュレーション画面はインタラクティブです：

| 操作 | 動作 |
|------|------|
| 左ドラッグ | 視点を回転 |
| スクロール | ズーム |
| 右ドラッグ | 移動 |
| Rキー | カメラをリセット |

---

## 📋 事前準備

- **Visual Studio Code** をノートPCにインストール — [ダウンロード](https://code.visualstudio.com/)
- **Remote - SSH** 拡張機能をVS Codeにインストール
- 接続情報はワークショップ当日に印刷カードで配布されます

---

## 困ったら

| 問題 | 解決方法 |
|------|----------|
| ライブ映像が見えない | VS Codeのポートタブで地球アイコンをクリック |
| 映像がフリーズした | Claudeに「ライブ配信を再開して」と伝える |
| Claudeが応答しない | `Ctrl+C` → `claude` で再起動 |

> 詳しくは [`docs/participant-guide.md`](docs/participant-guide.md) を参照してください。

---

<details>
<summary>📁 技術詳細（クリックで展開）</summary>

### 仕組み

- NVIDIA DGX Spark上の [MuJoCo](https://mujoco.readthedocs.io/) 物理エンジンでシミュレーションを実行
- ライブ映像はMJPEG形式でブラウザにストリーミング（VS Codeが自動でポート転送）
- [Claude Code](https://claude.ai/code) がPythonコードを生成・実行

### リポジトリ構成

```
physics-ai-workshop/
├── content/                    # シミュレーションモデル
│   ├── panda_ball_balance.xml  # 組み立て済みPandaアーム＋プレート＋ボール
│   ├── ball_and_plate.xml      # プレート＋ボール（スタンドアロン）
│   └── franka_panda/           # Franka Panda ロボットアーム
├── scripts/                    # ワークショップスクリプト
│   ├── 01_validate_assembly.py # スプリント1：ロボットを見る
│   ├── 02_pid_baseline.py      # スプリント2：PID発見
│   ├── 03_optimize_pid.py      # スプリント2：正解のPID検証
│   ├── 04_survival_map.py      # スプリント3+4：維持マップ
│   └── 05_challenge.py         # スプリント3+4：コントローラ探索
├── mujoco_streamer.py          # ライブ配信ヘルパー
├── docs/                       # ドキュメント
└── CLAUDE.md                   # AIエージェント用コンテキスト
```

### リソース

- [MuJoCo ドキュメント](https://mujoco.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Claude Code](https://claude.ai/code)
- [MuJoCo Pythonバインディング](https://mujoco.readthedocs.io/en/stable/python.html)

### ライセンス

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) — Franka Pandaモデルは [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Copyright Google DeepMind）より

</details>
