# nhc12-grasp-demo

Isaac Sim 上のロボットアームが、手首に取り付けたカメラで机上の赤いキューブを検出し、
その向きを算出してグリッパを合わせ、把持して持ち上げるデモである。

[English README](README.md)

| 項目 | 内容 |
|---|---|
| ロボット | Yaskawa Motoman NHC12 ＋ Robotiq 2F-85（単一の 12 自由度アーティキュレーション） |
| カメラ | Intel RealSense D455（手首搭載） |
| シーン | `scenes/arm310d_d455_ik_demo_r8.usd` |
| 対象物 | 赤い 50 mm キューブ（`/World/object`） |

---

## 1. クイックスタート

### 1.1 起動

1. Isaac Sim で `scenes/arm310d_d455_ik_demo_r8.usd` を開く。
2. **Play** を押す。物理シミュレーションが動作している必要がある。
3. **Script Editor** で `scripts/demo_start.py` を開き、`Ctrl+Enter` を押す。

コンソールに `{'status': 'ready', ...}` が表示されればロードは完了である。
この時点で IK 追従が有効になっており、ビューポートで `/World/IKTarget` を
ドラッグするとアームが追従する。

### 1.2 把持を 1 回実行する

キューブを机上の任意の位置に、任意の角度で置く。条件は、左カメラの画像に
写っていることだけである。

```python
demo_run(lift_mm=100)
```

各段階の進捗はコンソールに出力される。次の行が現れた時点で実行は完了である。

```
demo_run: DONE  outcome=held  object_rise_mm=100.45  elapsed_s=41.2
```

この行が出るまで次の操作を行ってはならない。統合された結果を見るには次を呼ぶ。

```python
demo_status()
```

### 1.3 解放と再実行

```python
demo_release()
```

キューブは机上に落下する。別の位置へ移し、任意の角度に回してから
`demo_run()` を再度呼べばよい。

---

## 2. アーキテクチャ

```
                    ┌─────────────────────┐
   D455 camera ────▶│  capture_*.py       │──▶ object position, size,
                    │  capture_approach.py│    and grasp axis
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    demo_api.py      │  ← the functions you call
                    └─────────────────────┘
                       │              │
              target pose             gripper angle
                       ▼              ▼
          ┌──────────────────┐  ┌───────────────────────┐
          │ /World/IKTarget  │  │ gripper_controller.py │
          │  position + yaw  │  └───────────────────────┘
          └──────────────────┘              │
                       │                    ▼
                       ▼            finger_joint (index 6)
          ┌──────────────────┐
          │ ik_controller.py │
          └──────────────────┘
                       │
                       ▼
              arm joints, index 0-5
```

> 図中のラベルは英語版と同一である。ソースコードおよび数式の同一性を保つため、
> 図・数式ブロックは翻訳しない。

関節角度を直接指令することはない。本デモは `/World/IKTarget` という座標フレームを
移動させ、`ik_controller.py` がツールセンターポイントを位置と姿勢の両方で
それに追従させる。グリッパは独立した経路であり、IK を経由しない。

**表 2-1　ファイル構成**

| ファイル | 役割 |
|---|---|
| `scripts/demo_start.py` | 起動入口。全モジュールを読み込み、IK 追従を開始する |
| `scripts/demo_api.py` | 公開 API。後述の関数はすべてここで定義される |
| `scripts/capture_d455.py` | 撮影の統括。レンダリング、保存、1 回分の結果の組み立て |
| `scripts/capture_approach.py` | 全景撮影、ハーフステップ接近、精密計測 |
| `scripts/capture_segment.py` | 平面フィッティングと連結領域分割 |
| `scripts/capture_geometry.py` | カメラ内部パラメータ、投影、姿勢生成 |
| `scripts/capture_annotate.py` | 注釈付き画像の描画 |
| `scripts/capture_result.py` | `result.json` / `diagnostics.json` の組み立て |
| `scripts/capture_run.py` | 実行ディレクトリ、ハッシュ、品質ゲート |
| `scripts/yaw_estimate.py` | 上面から把持軸と形状分類を算出 |
| `scripts/pose_math.py` | クォータニオン合成と対称性の畳み込み |
| `scripts/ik_controller.py` | IK 追従制御（アーム、index 0-5） |
| `scripts/gripper_controller.py` | グリッパ角度の指令と観測（index 6） |
| `scripts/grasp_demo.py` | 把持シーケンスと制御則の定数 |
| `outputs/captures/<run_id>/` | 撮影ごとの画像と計測結果 |

`yaw_estimate.py` と `pose_math.py` は Isaac Sim から何も import しない。
純粋な幾何計算であり、通常の `python3` で単体テストできる。

---

## 3. 動作原理

### 3.1 対象物の位置検出

カメラは手首に搭載されているため、机上のどの範囲が視野に入るかはアームの姿勢に依存する。
`demo_capture()` は次の 3 段階で動作する。

1. **全景撮影**　アームを高所の俯瞰姿勢へ移動させ、1 枚撮影する。この段階では
   `height_above_plane_m`（0.040〜0.060 m）のみで候補を選別する。机面からの高さは
   回転および部分的な見切れの影響を受けないため、対象物が画面内に一部しか
   入っていない場合でも利用できる。
2. **ハーフステップ接近**　候補が画像中心から離れている場合、中心へ寄せるのに必要な
   距離の半分だけカメラを移動させ、再度撮影する。これを `MAX_APPROACH_STEPS` 回まで
   繰り返す。
3. **精密計測**　対象物が中心付近に入った状態で、完全な判定基準を適用する。すなわち、
   高さ、水平方向寸法と高さの比（0.85〜1.50）、`touches_border`、
   `center_reliability` の 4 項目である。

接近ループには 2 つの安全機構がある。移動のたびに、対象物の画像中心からのピクセル距離は
前ステップより小さくなければならない。小さくならない場合、または候補が消失した場合は、
対象物が検出できた直近の姿勢へ復帰してループを終了する。

深度画素は世界座標へ逆投影され、机面は RANSAC で平面フィッティングされる（机面は
世界座標 z ≒ 0.595 m）。カメラから 0.18 m 以内の画素はロボット自身として除外し、
平面より上に残った点群を連結領域へ分割する。

### 3.2 対象物の向きの算出

平行グリッパは対象物を挟んで閉じる。したがって合わせるべきは「対象物の角度」ではなく、
**最も狭い方向**である。

各領域の上面を机面へ投影し、凸包を取り、**最小面積矩形**を回転キャリパー法で
求める。その矩形の短辺の方向が、指が跨ぐべき方向である。

```
grip_yaw_deg     = direction of the rectangle's short side
grip_width_mm    = length of the short side
symmetry_period  = 90° (square-like) / 180° (elongated) / none (round)
```

対称周期は計測された矩形から導出されるものであり、仮定してはならない。正方形は
等価な 4 方向のいずれからでも把持できる。長方形は正しい方向が一系統しかない。
円や円盤には優先方向が存在しない。

主成分分析は**意図的に用いていない**。正方形の共分散行列は等方的であり、
その固有ベクトルの方向は信号ではなく数値ノイズだからである。

> **理論**　平面フィルタリング、凸包、回転キャリパーの定理、充填率による形状判定、
> 対称性の畳み込み、および計算例を含む完全な導出は
> [`docs/README-yaw-estimate.ja.html`](docs/README-yaw-estimate.ja.html) に示す
> （[English](docs/README-yaw-estimate.html) · [日本語版](docs/README-yaw-estimate.ja.html)）。

### 3.3 アームの移動

`demo_trace()` は撮影結果から目標位置を算出する。

```
IK_target = surface_centre + [0, 0, -(object_height / 2) - 0.0021]
```

表面中心は対象物の上面中心であるため、目標点は高さの半分だけ下降して重心高さに達し、
さらに 2.1 mm 下がる。この 2.1 mm はグリッパ形状に由来する実測オフセットである。

姿勢は、下向きクォータニオンを世界 Z 軸まわりに `grip_yaw_deg` だけ回転させ、
さらに対象物自身の対称周期へ畳み込んだものである。これにより手首は常に短い側へ回る。
たとえば 135° という読み値を 90° 対称のもとで畳み込むと、135° 回転ではなく
−44.7° となる。

回転を適用するのはこの最終移動のときだけである。撮影の 3 段階はすべて固定姿勢で
行われる。カメラがグリッパに搭載されているため、撮影中に手首を回すと画像も一緒に
回り、ハーフステップの幾何が成立しなくなるからである。

### 3.4 把持

Robotiq 2F-85 は 6 個の関節を持つが、ドライブを備えるのは `finger_joint`
（アーティキュレーション index 6）のみである。残る 5 個には gearing ±1.0 の
PhysX ミミック拘束が設定されており、いずれも `finger_joint` を参照する。
したがって 1 関節を指令すればリンク機構全体が連動する。

- **0° が全開、47° が閉である。** 角度が大きいほど開口は小さい。
- ドライブは位置制御であり、`stiffness = 3.0`、`max_force = 26 N·m` である。

`demo_grasp()` は関節を 47° まで指令する。キューブに阻まれて実際の角度は 21.5° 付近で
停止し、指令値と実現値の差がモータトルクを力上限まで蓄積させる。このトルクが把持力である。
閉動作中に `tracking_error_rad` が増大するのは把持力の指標であり、異常ではない。

---

## 4. 関数リファレンス

**表 4-1　公開関数**

| 関数 | 動作 |
|---|---|
| `demo_run(lift_mm=100.0)` | 撮影・移動・把持を 1 つのタスクとして実行する。段階ごとの進捗と、最後に `DONE` / `FAILED` の 1 行を出力する |
| `demo_capture()` | 全景撮影、接近、計測を行う。結果は `demo_status()` で取得する |
| `demo_trace(surface_center_xyz=None, thickness_m=None)` | グリッパを全開にし、手首を把持軸へ回転させ、IK ターゲットを対象物中心へ移動する |
| `demo_grasp(lift_mm=100.0)` | グリッパを段階的に閉じ、持ち上げる |
| `demo_release()` | グリッパを 0° へ開き、記録された trace 目標を消去する |
| `demo_status()` | `stage`、`running`、`ok`、`result`、`error` を返す |
| `demo_gripper_angle(deg)` | フィンガー関節を直接指令する（0〜47°） |
| `demo_gripper_status()` | 現在のグリッパ角度と追従誤差を返す |

`demo_capture()`、`demo_trace()`、`demo_grasp()`、`demo_run()` は非同期タスクを
スケジュールして即座に戻る。Isaac Sim の asyncio ループは Kit の update イベントで
駆動されるため、Script Editor 内でブロッキング待機を行うとループが再入し、
そのスケジューリングが中断される。`demo_run()` は 1 つのタスク内部で各段階を
`await` するため、呼び出しは 1 回で足りる。

### 4.1 手首の位置合わせに関する項目

`demo_trace()` は手首をどう回転させたかを報告する。

**表 4-2　回転の報告項目**

| 項目 | 意味 |
|---|---|
| `yaw_applied_deg` | 対称性の畳み込み後に実際に適用された回転量 |
| `yaw_source` | `capture` / `indeterminate` / `none`。下表参照 |

**表 4-3　`yaw_source` の値**

| 値 | 条件 | `yaw_applied_deg` |
|---|---|---|
| `capture` | 通常。撮影結果から読み取れた場合 | 畳み込み後の把持軸 |
| `indeterminate` | 対象物が円形であり、回転が無意味な場合 | `0.0` |
| `none` | 項目自体を読み取れなかった場合 | `0.0` |

`none` の場合は必ずコンソールに出力される。無言の代替値ではない。

### 4.2 把持結果の分類

`demo_grasp()` は `outcome` フィールドを返す。

**表 4-4　`outcome` の値**

| 値 | 意味 |
|---|---|
| `held` | 対象物がアームとともに上昇した（指令値の 5 % 以内） |
| `ejected` | 対象物がアームより高く上昇した。指で弾き出された状態である |
| `partial_slip` | 上昇量が指令値より 5〜15 % 少ない |
| `dropped` | 上昇量が指令値より 15 % を超えて少ない |
| `no_contact` | 接触が検出されなかった。持ち上げ動作は実行されない |

`slip_mm` は後方互換のために残されている。絶対値を用いるため `ejected` と `dropped`
を区別できない。判定には `outcome` を用いる。

---

## 5. エラー

以下は同期的に送出され、Script Editor にトレースバックとして表示される。

**表 5-1　主なエラーと対処**

| メッセージ冒頭 | 原因 | 対処 |
|---|---|---|
| `gripper not fully open` | グリッパが対象物を保持したままである | `demo_release()` を呼ぶ |
| `most recent capture failed` | 直近の撮影が成功していない | `demo_capture()` を再実行する |
| `no demo_trace() target recorded` | `demo_trace()` が未実行、または `demo_release()` により消去された | `demo_trace()` を呼ぶ |
| `IK target drifted from the traced pose` | `demo_trace()` 後に IK ターゲットが移動した | `demo_trace()` を再実行する |
| `no cube candidate among N objects` | 判定基準に合致する領域がなかった | メッセージ中の注釈画像を確認する |

ドリフトのメッセージには、位置と姿勢のどちらがどれだけずれたかが明記される。
失敗時のメッセージには注釈画像のパスと、各候補が抵触した規則が列挙される。
座標範囲は含まれない。

---

## 6. 撮影画像の保存先

`demo_capture()` は実行ごとに `outputs/captures/<run_id>/` を生成する。

**表 6-1　生成されるファイル**

| ファイル | 内容 |
|---|---|
| `rgb_left.png`、`rgb_right.png`、`rgb_color.png` | D455 の 3 センサの生画像 |
| `rgb_left_annotated.png` | 検出領域、照準、表面中心の注釈付き画像 |
| `region_overlay.png` | 領域分割の重ね合わせ図 |
| `depth_preview.png`、`depth_left_annotated.png` | 深度の可視化画像 |
| `depth_axial_left.npy`、`depth_radial_left.npy` | float32 の生深度配列 |
| `result.json`、`diagnostics.json` | 計測結果と診断項目 |

`result.json` の各対象物は `grip_yaw_deg`、`grip_width_mm`、`object_length_mm`、
`symmetry_period_deg`、`shape_class`、`fill_ratio` を持つ。
旧項目 `yaw_deg_estimated` は互換のために残されているが、90° の剰余であり
正方形以外を記述できない。判定には `grip_yaw_deg` を用いる。

`outputs/` は `.gitignore` に登録されているため、`git status` には現れず、
クローン直後にも存在しない。ディレクトリを直接開いて確認する。
出力先を変更する場合は環境変数 `D455_DEMO_OUTPUT_DIR` を設定する。

---

## 7. 適用範囲と制限

**対応範囲**　
左カメラに写る範囲であれば、机上の任意の位置に置かれた赤い 50 mmキューブを、鉛直軸まわりの任意の回転角において扱える。グリッパは閉じる前にキューブへ自動的に位置合わせを行う。

**運用上の注意**
次の `demo_run()` の前に `demo_release()` を呼ぶこと。グリッパが保持したままのキューブは机上に存在せず、次の撮影で検出できない。
