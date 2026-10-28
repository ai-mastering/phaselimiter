# FFT比較実験

目的は **速度** と **理論値に対する誤差** の比較。x86＋IPPは誤差比較の
基準として使い、Apple Siliconとの速度比較には使わない。

本ディレクトリは独立したCMakeプロジェクト。本体のソース、ヘッダ、CMake、
サブモジュールを参照しない。追加依存、ダウンロード、生成物は既定で
`.work/`以下に置き、Gitには含めない。外部に追加する設定は
`.github/workflows/fft-experiment.yml`だけ。

## 比較対象

| 実装 | 役割 | 任意長への対応 |
|---|---|---|
| Accelerate | 現行実装の問題を調べる特別枠 | 対応長だけ。非対応は明示的にskipし、別実装へ逃がさない |
| IPP | x86の誤差比較基準 | 実数／複素数DFT API |
| FFTW | 性能・精度の比較基準、採用条件は別途検討 | 任意長 |
| PocketFFT | ポータブルな採用候補 | 実数／複素数とも任意長 |
| KissFFT | 小さなポータブル実装の候補 | 複素数は任意長。奇数の実数入力も同じKissFFTの複素数APIで処理 |

float、double、実数、複素数を別々に測る。ライブラリ×精度ごとに独立した
実行ファイルを作り、FFTWとIPP等を同じ実行ファイルへリンクしない。
依存のバージョン／コミットとSHA-256は`dependencies.json`で固定する。
FFTWはARM NEONに対応した上流autotoolsビルドを使う。FFTWのCMakeビルドへ
置き換えるとSIMDの条件が変わるため注意。

## 精度の検証

`fixtures.py`はPython標準ライブラリのDecimalを80桁精度で使い、FFT
ライブラリに依存せず共通データを生成する。

- ゼロ、DC、ナイキスト、先頭／途中／末尾のインパルス、端点＋インパルスの混合。
- 小サイズの疑似乱数入力は直接DFTで全出力を計算。
- 大サイズも解析解を持つ入力について**全出力**を比較し、一部binの抽出で済ませない。
- 順変換の出力とは独立した疎なスペクトルから逆変換を検証。
- 往復変換、FFT→スペクトル乗算→逆FFTによる循環畳み込みを検証。
- DC／ナイキストの誤差を個別に記録。同じ演算を2回行い完全一致するかも記録。

入力はfloat32でも厳密に表せる2進小数を使う。これにより入力の量子化誤差を
ライブラリ差と混同しない。参照値は80桁で計算した後、交換用にbinary64へ
丸める。したがって**doubleの参照値には最大約0.5 ULPの保存時丸めがある**。
80桁のまま誤差を計算する試験ではない。各実装の出力もbinary64で保存する。

既定の28種類の長さには1／2、小さな奇数・素数、255／256／257、1000／1024、
4095／4096／4097、16384／16385を含む。奇数長にはナイキストbinが存在しない
ため、その信号ケースだけを省く。奇数長そのものは検証する。

記録するのは最大絶対誤差、その位置、相対L2誤差、参照値の最大振幅、端点誤差。
誤差はbin数や振幅によって変わるので、異なるケースを集約した最大値だけで
ライブラリの優劣を判断しない。IPPとの差も記録するが、**IPPとの一致を正解とはしない**。

非有限値、非再現、粗いsanity boundを超える誤差は失敗にする。
sanity boundは`64 * epsilon * max(1, log2(N)) * max(1, reference_max)`。
これは正規化や配置の大きなミスを検出する補助条件で、許容精度の保証ではない。
通過していても誤差の実測値はすべて保存する。本体のテストや許容誤差は変更しない。

## 速度の検証

- 全ライブラリをビルドしてから、同じランナー内で順番に実行する。
- 順変換と逆変換は別々に計測。プラン作成時間も別レコードに残す。
  プラン作成は各長さ・domainで1回の観測値なので、統計的な速度保証には使わない。
- プラン・作業領域は再利用し、ウォームアップ後にバッチ回数を調整して7試行。
  最小値、中央値、各試行の時間を保存する。
- 計測対象は共通APIの呼び出し。各アダプターの入力コピー・配置変換を含む。
  ネイティブのFFTカーネルだけの測定ではない。逆変換の計測には`1/N`の正規化を含まない。
- ライブラリ内部の作業領域確保が実行API内にある場合、そのコストも含む。
- FFTWは既定で`MEASURE`。`--fftw-planner estimate`でも試せる。FFTW自身の
  プランニング時間も作成時間に含む。単一スレッドで使う。
- PocketFFTは固定したコミットの1DプランAPIを使う。内部API依存はこの実験用
  アダプターだけに置く。KissFFTも単一スレッド。
- IPPとQEMUでは実行時間を測らない。同じ実行、同じ精度・長さ・方向・domainの
  結果だけで速度比を出す。別ホストの速度比は出さない。

GitHub共有ランナーの時間にはノイズ、コア割り当て、実行順や温度の影響がある。
性能の閾値でCIを落とさず、複数回の測定で傾向を見る。採用候補が絞れたら、
本体のデータ配置と処理全体を含めた性能検証を別途行う。

## ローカルで実行

Python 3.10以上、CMake 3.23以上、Ninja、C/C++17コンパイラ、make、ネット接続が必要。
Pythonの追加パッケージは不要。初回だけ固定した依存ソースを取得する。

```sh
python3 -m unittest discover -s experiments/fft -p 'test_*.py' -v
python3 experiments/fft/run.py --backends pocketfft,kissfft,fftw
python3 experiments/fft/report.py \
  --results experiments/fft/.work/results \
  --output experiments/fft/.work/summary
```

Linuxの既定ビルド並列数は`nproc`。`run.py`は
`cmake --build ... --parallel <並列数>`を使う。`--jobs`で指定もできる。
候補を一つだけ試す場合は`--backends pocketfft`、短い正しさの確認には
`--sizes 1,2,3,5,16,31,257,1024 --timing-ms 0`を指定する。
結果を保持する場合は`--output-dir experiments/fft/.work/results-名前`を指定する。
速度を測る長さの既定値は256、1000、1024、4096、4097、8192、16384。
`--timing-sizes 512,2048`や`--timing-sizes all`で変更できる。
同じ出力ディレクトリを再利用すると該当ライブラリの結果を上書きする。
異なる長さや参照データの結果同士は比較できない。

macOS ARMでは`--backends pocketfft,kissfft,fftw,accelerate`。
Linux x86でIPPも含める場合は`--backends pocketfft,kissfft,fftw,ipp`。
IPPが未指定なら固定した2021.9.0のdebを`dpkg-deb`でローカル展開する。
既存のIPPを使う場合は`--ipp-root /path/to/ipp`で指定する。

### Linux ARMをQEMUで実行

Debian/Ubuntuの例：

```sh
sudo apt-get install qemu-user gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
python3 experiments/fft/run.py --qemu --fftw-planner estimate \
  --backends pocketfft,kissfft,fftw \
  --output-dir experiments/fft/.work/qemu-results
```

同じ参照データとテストを使い、ARM64命令をQEMUで実行する。エミュレーションは
Apple Siliconそのものではなく、macOS／AppleClangの差も再現しない。
Accelerateは実行できない。ARMの正しさをローカルで反復確認し、実機の確認は
Actionsで継続する。QEMUでFFTWの`MEASURE`を使うとエミュレーター向けの
計画を作るだけなので、既定手順では`ESTIMATE`を指定する。

## GitHub Actions

専用の **FFT experiment** ワークフローを手動実行できる。
新規ワークフローの手動実行はGitHubの仕様上、既定ブランチに入った後に利用できる。
本ディレクトリかワークフローの変更時のpush／pull requestでも実行する。
既存の本体CIは変更しない。サブモジュールの取得も不要。

1. macOS ARM：4実装を同じランナーで測定。
2. Linux x86：3実装を測定し、IPPの誤差基準も生成。
3. Linux x86＋QEMU：3実装のARM64版を正しさだけ検証。
4. 結果を集約し、各実装とIPPの全出力を比較。同じネイティブ実行内では
   PocketFFTを基準にした出力比較も残す。

artifactにはJSONL、出力値のbinary64スナップショット、環境・依存情報、
ビルド／実行ログ、Markdown／JSONの集計を残す。実行ファイルや依存ライブラリ、
SDKは公開しない。`summary.md`をActionsのSummaryにも表示する。
誤差の詳細は各実装の`results.jsonl`と集約した`summary.json`にある。
`reference.json`のハッシュが違う実行は比較を拒否する。

環境情報にはCPU、OS、アーキテクチャ、コンパイラ、ライブラリの版、
FFTWプランナー、native／QEMUを記録する。`configure.log`／`build.log`には
実際のビルド設定を保存する。

## 元のトゥルーピーク失敗を調べる診断

`run_peak_probe.py`はFFT単体の比較とは別に、元の`AudioAnalyzerPeak.Impulse`の
条件を再現する。失敗値`-5.1771939979516901e-07 dB`は、振幅が`1.0f`の直前の
float値`0.999999940395355224609375`になった場合のdB値と完全に一致する。
この一致だけでは、丸め誤差と処理ミスの区別はつかない。

```sh
# Linux x86。既存のIPPがあれば --ipp-root も指定できる。
python3 experiments/fft/run_peak_probe.py --backends ipp,pocketfft,fftw,kissfft

# macOS ARM
python3 experiments/fft/run_peak_probe.py --backends accelerate,pocketfft,fftw,kissfft
```

元の44,100サンプル、4倍オーバーサンプリング、4,097係数、16,384点FFT、
16,384サンプル単位の外側処理、内部の12,288＋4,096分割、全末尾処理を再現する。
係数は元と同じdouble生成・double FFT後にfloat化する経路と、80桁参照の
スペクトルをfloat化する経路、係数FFT自体もfloatで行う経路を比較する。
実数FFTと、実数入力を複素数FFTで処理する経路を別々に実行する。
呼び出し側の係数生成・スペクトル乗算は`-ffp-contract=off`／`on`を別の
実行ファイルで比較する。`on`は積和融合の許可であり、必ず融合するという意味ではない。
FFTバックエンドは既存の実験と同じ設定を維持する。`--contracts off`で片方だけにもできる。

本体のソースには依存しない。FIRのスカラー式とoverlap-addの更新順序は
コミット`14b5027`の固定した再現ケースとして実験側に置く。
`reproduction/dbesi0.cpp`は同コミットの大浦氏のI0実装を、その許諾表示とともに
保存したもの。本体に追従する別実装としては保守しない。
独立した理論側はDecimalのI0級数とsinc/Kaiser式を使う。

各ホストが生成した実際のbinary64係数を80桁の独立FFTへ入力し、DC／ナイキストを
含む7 binでは直接DFTとも照合する。こうして、係数生成の差とFFTの差を分ける。
実測スペクトルの直接逆変換和から、頂点の差を係数スペクトルのFFT／float化、
順FFT、乗算、逆FFTに分解する。直接和はlong doubleで計算し、位相の参照は
80桁で生成してbinary64へ保存するため、double相当以下の微小差を保証する検証ではない。
long doubleの仮数ビット数も記録する。

インパルス位置、処理単位を変え、次の独立検証も行う。

- 元と同じoverlap-addバッファと、独立した出力への直接加算を全サンプルで比較。
- DCとナイキストを先頭の1複素数スロットへ詰めるPerm演算と、個別bin演算を比較。
- DC欠落、ナイキスト欠落、両者を通常の複素数として乗算するミスを意図的に注入。
- ナイキストがほぼゼロの元のローパスに加え、DC利得1／ナイキスト利得0.5の
  `[0.75, 0.25]`を4,097係数へゼロ詰めしたフィルターでも対照実験。

元の厳しいdBアサーションを通るかは**観測値**として保存し、その失敗だけでは
診断ジョブを落とさない。構造比較の不一致や対照実験の検出失敗はジョブを落とす。
許容誤差は本体でも実験の既存テストでも変更しない。

`peak-summary.md`／`peak-summary.json`に集計と全診断値を保存する。
元の入力・処理順序で得た波形も`original_wave.bin`へ保存する。
専用Actionsのネイティブジョブでも実行し、同じartifactとSummaryに含める。
QEMUで試す場合は`--qemu`を指定できる（ポータブル実装のみ）。

初期のLinux x86／IPP 2021.9.0の確認では、本体ヘッダの関数を直接呼んでも、
元のインパルス位置22,050では0 dB、位置44,099では元のmacOS失敗と同じ
`-5.1771939979516901e-07 dB`、位置1では`-1.035438799590338e-06 dB`になった。
固定再現ケースでも同じ値を得た。また、FFTW 3.3.11のfloat実数FFTをESTIMATEで
使うと、元の位置22,050のまま`-5.1771939979516901e-07 dB`を再現した。
これは「IPPで通った1点の結果」を
位置に依存しない精度保証とみなせないことを示すが、Accelerate側の個別の
誤差段階はmacOSでの診断結果を見て判断する。

## ライセンス

実験のコードはリポジトリのMITライセンス。依存ライブラリの条件は別途適用される。

- PocketFFT、KissFFT：BSD-3-Clause。ソースの著作権・条件・免責を保持し、
  バイナリを配布する場合も必要な表示を同梱する。
- FFTW：GPL-2.0-or-later。GPL条件でリンク済み実行ファイルを配布する場合は、
  そのプログラムのGPL条件と対応ソースの提供に対応する必要がある。
  動的リンクでも回避できない。商用ライセンスは別途。
- IPP：取得するパッケージに付属するIntelライセンスが適用される。
- Accelerate：AppleのOS／SDKの利用条件が適用される。

本体にはリンクせず、測定結果・ログだけをartifactにする。
本体への採用・バイナリ配布は、性能・精度の結果とライセンス条件を確認してから
別の変更として行う。依存のライセンス本文は取得したソース／パッケージにも含まれる。
