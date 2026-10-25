# GradCoreのAVX-512調査メモ

2026-10-25、対象コミット: `b126463503c0531f28a92e35af0f8e5068caf1be`。
実装は未変更。一時コピーでの測定では、16要素化によりシングルスレッドの
GradCalculatorが約1.14〜1.17倍になった。リミッター全体・収束・最終WAVは未測定。

## 条件と結果

- CPU: Ryzen 9 9950X、コンパイラ: GCC 14.4.0。
- 共通: `-O3 -DNDEBUG -march=znver5 -mtune=znver5`、fast-mathなし。
- AVX2・8要素: `-mno-avx512f -mprefer-vector-width=256`。
- AVX-512・8/16要素: `-mprefer-vector-width=512`、libsimdppに
  `SIMDPP_ARCH_X86_AVX512F` / `DQ` / `BW` / `VL`を追加指定
  （後3つも同じ`SIMDPP_ARCH_X86_AVX512`接頭辞）。AVX2/FMA3の指定は維持。
- 共通のbakuageは`-march=core-avx2 -mtune=core-avx2`で一度だけビルド。
  全構成で同じIPP 2021.9、oneTBB 2023.1を使用。
- `taskset -c 2`、`tbb::global_control::max_allowed_parallelism=1`、
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。
- 各構成を順番をローテーションして6回実行し、初回を除く5回の実時間中央値。
  Google Benchmarkの各ケースは`--benchmark_min_time=0.12`。

GradCalculatorの既存ベンチマーク（`waveLen=44100*10`のインターリーブ配列、
ステレオ5秒相当、引数`/1`のみ）の結果:

| 処理 | ソースFFTキャッシュ | AVX2・8要素 | AVX-512・8要素 | AVX-512・16要素 | 16要素/AVX2の速度比 |
|---|---|---:|---:|---:|---:|
| 評価値 | なし | 3.751 ms | 3.742 ms | 3.227 ms | 1.163倍 |
| 評価値＋勾配 | なし | 6.278 ms | 6.278 ms | 5.350 ms | 1.173倍 |
| 評価値 | あり | 3.545 ms | 3.563 ms | 3.107 ms | 1.141倍 |
| 評価値＋勾配 | あり | 6.122 ms | 6.196 ms | 5.307 ms | 1.154倍 |

GradCore単体（FFT長256/1024/4096/16384）は、AVX2版比でキャッシュなし
1.14〜1.21倍、あり1.16〜1.33倍。AVX-512設定だけの8要素版はほぼ同速なので、
主な改善は16要素化による。16要素版の生成コードでZMMの演算命令も確認済み。

## 再開手順

一時成果物は`/tmp/grad-width-bench/`にある。残っていれば
`bash /tmp/grad-width-bench/reproduce.sh`で再実行できる。
`summary.csv`、`validation.json`、`compile-commands.txt`、`environment.json`と
変更済みコピー`source/`を参照。一時ファイルの永続性は保証されない。

一時成果物がなくなっていたら、上記コミットを別ディレクトリへコピーし、次を再現する。

1. `src/phase_limiter/config.h`の`DefaultSimdType`を、構成ごとに
   `simdpp::float32x8`または`simdpp::float32<16>`にする。
   **同梱版には`simdpp::float32x16`という別名はない。**
2. `src/bench/grad.cpp`と`grad_calculator.cpp`から3構成の測定用実行ファイルを作り、
   `phaselimiter_cli`と`benchmark::benchmark`にリンクする。
   `src/bench/utils.h`の`MY_SIMD_BENCH`は`DefaultSimdType`だけ登録する。
   測定ループの戻り値を`benchmark::DoNotOptimize(...)`に渡す。
   未使用の評価値が最適化で消える測定結果は採用しない。
3. 測定用mainでTBBを1スレッドに制限し、
   `GradCoreSettings::GetInstance().set_src_cache(false/true)`を切り替える。
   `--benchmark_filter='/(256|1024|4096|16384|1)$'`で測る。
4. 上記コンパイラ設定を使う。`cmake/ProjectOptions.cmake`のINTERFACE指定が
   ターゲット指定より後に付いて`-march=core-avx2`へ戻す場合があるため、
   コピー側で競合を解消し、`compile_commands.json`で実際の引数順を確認する。
   ローカルビルドは`cmake --build <build> --parallel "$(nproc)"`。

## 正しさと未解決点

全帯域指定（`max_available_freq=44100`、sample_rate=44100）の24ケースでは、
8/16要素の勾配は完全一致、評価値の最大相対差は`1.624e-6`だった。
検証条件はFFT長4種類×キャッシュ有無×入力3種類（ゼロ、独立乱数、
参照波形に微小差分）。乱数は`mt19937(123)`、一様分布[-1,1]、
微小差分の倍率`1e-5`、noise=`0.01`。

ただし、`GradCore.h`の`nonZeroSpecSrcLen`は`2 * SimdType::length`単位で
切り上げるため、**周波数上限で2つの計算ループを分ける境界が幅依存**。
上限を12000 Hzとし、それより上にも成分のある同じ乱数入力を与えると、
評価値の最大相対差約8.4%、勾配の最大相対L2差約26.9%を観測した。
上限より上がゼロという想定に反する入力での結果であり、通常音源で同じ差が
出ることを意味しないが、型変更を常に等価とは扱えない。

次はこの境界の仕様を決め、帯域制限音源・内部オーバーサンプリング・端数長で
正しさを確認し、その後リミッター全体の速度と収束・出力差を比較する。
コンパイル時選択か実行時ディスパッチかは未決定。
なお`FastRcp`/`FastSqrt`の8要素特殊化は現行呼び出し側が`#if 0`で無効。
