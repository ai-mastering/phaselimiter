# GradCore FFT置換研究の再開メモ

2026-10-25、対象コミット `b126463503c0531f28a92e35af0f8e5068caf1be`。
実装変更は未採用。研究用コピーは `/tmp` にあり、本メモはその再開用。
[GradCoreのSIMD幅変更](gradcore-avx512.md)とは別実験であり、速度比を掛け合わせてはいけない。

## 現在の結論

9950X（Zen 5）、1スレッド、44.1 kHz・10秒ステレオの実際の
`GradCalculator::calcEvalGrad`（評価＋勾配、窓掛け・重ね合わせ・タスク処理込み）:

| 構成 | IPP対照 | 置換後 | 時間短縮 |
|---|---:|---:|---:|
| 256〜16384点を選定済み自前FFTへ置換 | 13.17〜13.21 ms | 11.34〜11.46 ms | 13.3〜13.9% |
| 256〜8192点を置換、16384点はIPP | 13.16〜13.18 ms | 11.25〜11.30 ms | 14.3〜14.6%（約1.17倍） |
| 256〜8192点の逆変換だけ置換 | 13.16〜13.17 ms | 11.69〜11.74 ms | 10.9〜11.1%（約1.12倍） |

範囲は乱数／トーン＋ノイズの2入力の結果で、信頼区間ではない。
16384点は単独FFTの勝ちが実処理に持ち越せず、IPPへ戻した。
ソースFFTキャッシュは既定のOFF。ONでは後2構成とも約10%短縮。
外側のFISTA・ラインサーチ、I/O、収束回数、最終音質は未検証。
初期バイナリでは全置換の短縮が約10%だった。最終表は同一バイナリ内の対照比較だが、
コード／メモリ配置の影響があり、常に14〜15%とは限らない。

**現時点の採用候補は逆変換だけ。** 順FFTの丸め差が小さいnoiseで増幅される。
`noise=1e-6`の定数入力で、全置換のIPP比評価値差は最大約6.76%、
勾配の相対L2差は約1.289（129%）。順変換だけの置換でも再現する。
IPPが数学的な真値という意味ではないが、順変換置換の等価性は未確立。
`noise=0.01`でもDC入力では差が出るので、通常条件なら問題なしとは言えない。
逆変換だけの統合試験では評価値は完全一致し、勾配の相対L2差は約2e-7以下
（`noise=1e-6`を含む）。統合部分のASan/UBSanも通過。
プリビルドFFT・IPP・TBB内部はそのサニタイザの対象外。

## 計算契約と選定カーネル

- 44.1 kHzのGradCalculatorは256〜16384点の2冪、2ch、50%オーバーラップ。
  128点は単体試験のみ。キャッシュOFFの勾配は窓ごとに順FFT×2＋逆FFT×1、
  ONでウォーム済みなら順FFT×1＋逆FFT×1。評価だけなら逆FFTなし。
- 元と同じ事前計算済みfloat32窓（Hann / sqrt(N)）を使う。
  順変換は窓掛けを融合してCCSのパディングも初期化。
  逆変換はDC補正済みのPerm形式を受け、窓掛けと既存勾配への加算を融合する。
  正規化、CCS/Perm配置、DC/Nyquistの扱いを変えない。
- 長さごとのプランと一時領域をスレッドローカルで固定確保。
  実験ではGradCoreへ勾配出力先を渡し、GradCalculatorの既存ゼロ初期化・
  タスク順序を保った。float32x8・oversample=1だけを統合検証した。

以下のパスは `/tmp/` 起点。共通API・ヘッダも必要で、cpp単独では再構築できない。

| 長さ | 順変換 | 逆変換 |
|---|---|---|
| 256〜4096 | `gradcore-fft-window/selected/c{N}.cpp` | 同左 |
| 8192 | `gradcore-fft-realblocks/t4l32p2h3u1.cpp` | `gradcore-fft-l1/fw8.cpp` |
| 16384（実処理では不採用） | `gradcore-fft-realblocks/k1b0_16384.cpp` | `gradcore-fft-l1/fw0.cpp` |

探索済み: SIMD codelet、stage fusion、実数FFTの分解、転置／アドレスへの並べ替えの
織り込み、DIT/DIF・Stockham、AVX2/512、2,8 split-radix等。
直近では逆変換の窓テーブルをアクセス順に並べる変更が8192点で約2%改善。
YMM化で列用scratchを64→32 KiBへ縮小する案、twiddle漸化式、追加fusionは遅かった。
L1容量だけで性能を説明しない。devcontainerの `perf_event_open` はEPERMで、
性能カウンタによる原因確認はできていない。

## 再現と成果物

入口: `/tmp/gradcore-fft-endtoend/`。

- `report.txt`: 詳細な条件・失敗例・測定上の制限。
- `integration.patch`, `prepare.py`, `original/`, `custom/`: 統合差分と元／変更コピー。
- `backend.cpp`, `measure.cpp`, `numeric.cpp`: 実処理比較、数値検証、順逆の原因切り分け。
- `kernels/CMakeLists.txt`, `kernels/dispatch.cpp`: カーネル依存とサイズ別選択。
- `hybrid-summary.csv`, `safe-summary.csv`: 最終比較。`safe`は逆変換のみという
  実験名で、安全性の保証ではない。`numeric.csv`, `stress.log`に数値問題を保存。
- `/tmp/gradcore-fft-endtoend-code-and-results.tar.gz`: 統合実験のコードと結果の控え。
  これだけでは依存する過去のカーネル・SDK・ツールチェーンは復元できない。

```sh
bash /tmp/gradcore-fft-endtoend/reproduce.sh
```

このスクリプトはカーネルと比較器をビルドし、通常／低noise検証、順逆切り分け、
各構成3プロセスの測定を実行する。ローカルビルドは
`cmake --build ... --parallel "$(nproc)"`。測定とビルドを同時実行しない。

必要な環境:

- 非FFT部分: GCC 12.2、`-O3 -march=core-avx2 -mtune=core-avx2`、
  AVX2/FMA3、fast-math追加なし。元／変更版を別namespaceで同一実行ファイルにリンク。
- FFT: `/tmp/gradcore-fft-expansion/vendor/gcc-new/bin/x86_64-buildroot-linux-gnu-g++`
  （GCC 15.3）、`-O3 -ffast-math -march=znver5 -mtune=znver5 -mprefer-vector-width=512`。
- IPP: `/opt/phaselimiter/opt/intel/oneapi/ipp/2021.9.0`。
  TBB: `/tmp/onetbb-upgrade/sdk/opt/intel/oneapi/tbb/2023.1`。
- 論理CPU6固定、TBB並列数・calculator workerCountとも1。
  同じ配列を反復。プラン生成・入力コピー・検証は時間外。
  10秒は**各ch 441000フレーム**。noise=0.01、ERB重みなし、全帯域指定。
  ウォームアップ後15ブロック、元／変更版の順序を交互にし、3プロセスの中央値を採用。

過去の比較・探索は `/tmp/gradcore-fft-{research,window,layout,stockham,realblocks,l1}`。
`research` にIPP・oneMKL・FFTW・AOCL（FFTW/FFTZ）の比較がある。
**これらは一時ファイルで永続性はない。再開前に存在を確認し、残っていれば
ヘッダ・生成器を含む依存ディレクトリごと退避する。** 消失していた場合、本メモだけで
同一カーネルを完全復元することはできず、上の計算契約から再実装・再測定が必要。

次は逆変換のみの構成を基準に、実音源と最適化ループ全体で速度・収束・出力差を確認する。
順変換はDC／低noiseの誤差増幅を先に切り分ける。単独FFTの速さだけで採否を決めない。
