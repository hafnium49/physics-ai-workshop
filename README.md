# FY2026 物理AIワークショップ

AIコーディングエージェントでデジタルツインを構築

## 概要

素材エンジニアが [Claude Code](https://claude.ai/code)（AIコーディングエージェント）を使い、NVIDIA DGX Spark上でMuJoCo物理シミュレーションを構築・最適化する1時間のハンズオンワークショップです。

プログラミング経験は不要です。AIにやりたいことを日本語で伝えるだけで、AIがコードを書いてくれます。

## 構築するもの

**Franka Panda ロボットアーム**がプレートを持ち、その上にボールを載せたシミュレーションを構築します。目標：Claude Codeを使ってシミュレーションを組み立て、物理挙動をテストし、PID制御パラメータを最適化して、ボールを10秒間バランスさせることです。

## ワークショップ構成

| スプリント | 時間 | 目標 | 内容 |
|--------|------|------|------|
| 1. 探索 | 15分 | ロボットを理解する | 構築済みシミュレーションを実行し、関節を動かして直感を養う |
| 2. PID発見 | 12分 | ロボットにバランスを教える | 動作しないコントローラを実行し、Claudeに診断・修正を依頼する |
| 3. 初回イテレーション | 8分 | 改善ワークフローを体験 | ベースラインスコアを測定し、最初の改善を試して効果を確認する |
| 4. 自由探索 | 25分 | コントローラを改善する | サバイバルマップをスコアボードとして使い、Claudeにより良い制御戦略を試してもらう |

## 事前準備

- **Visual Studio Code** をノートPCにインストール済みであること
- **Remote - SSH** 拡張機能がVS Codeにインストール済みであること
- 接続情報はワークショップ当日に印刷カードで配布されます

## リポジトリ構成

```
physics-ai-workshop/
├── content/                # シミュレーションモデル
│   ├── ball_and_plate.xml  # プレート＋ボール（バランス対象）
│   └── franka_panda/       # Franka Panda ロボットアーム（MuJoCo Menagerieより）
├── docs/
│   ├── participant-guide.md    # ワークショップ参加者向けステップバイステップガイド
│   └── host-preparation-runbook.md  # ワークショップ主催者・管理者向け
└── CLAUDE.md               # Claude Codeセッション用のコンテキスト
```

## クイックスタート

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

## シミュレーションコンテンツ

### Franka Panda アーム

[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) から取得した7自由度ロボットアームです。MuJoCo Menagerieは、Google DeepMindが管理する高品質なオープンソースロボットモデルのコレクションです。正確なメッシュ、関節制限、動力学パラメータが含まれています。

### ボールとプレート

フラットなプレートとフリージョイントを持つボールを定義した最小構成のモデルです。ワークショップの課題は、これをPandaのグリッパーに取り付け、PID制御でボールをバランスさせることです。

## リソース

- [MuJoCo ドキュメント](https://mujoco.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Claude Code](https://claude.ai/code)
- [MuJoCo Pythonバインディング](https://mujoco.readthedocs.io/en/stable/python.html)

## ライセンス

このワークショップリポジトリは [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) ライセンスの下で公開されています。Franka Pandaモデルは [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache 2.0、Copyright Google DeepMind）から取得しています。
