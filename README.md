# Estimation of Oral Breathing

このプロジェクトは、2023年度卒業制作として開発された、遠赤外線動画を用いてヒトの口呼吸を推定するWebアプリケーションのソースコードです。

## 目次

- [特徴](#特徴)
- [実行環境](#実行環境)
- [設定](#設定)
- [使用方法](#使用方法)
- [参考研究](#参考研究)
- [連絡先](#連絡先)

## 特徴

- 動画解析の処理負荷が大きいので、解析結果を返す箇所をAPI `/analysis` として実装

## 実行環境

- Python 3.8+
- 必要なPythonパッケージ(requirements.txtに記載)

## 設定

1. プロジェクトのルートディレクトリに `.env` ファイルを作成し、以下の環境変数を追加します(具体的な数値が必要であれば、[下記連絡先](#問い合わせ)までご連絡ください):

    ```env
    TEMP_R2=0123456789
    TEMP_INTERCEPT=0123456789
    VOL_R2=0123456789
    VOL_INTERCEPT=0123456789
    ```

2. `inf_calibration`ディレクトリと`vis_calibration`ディレクトリに、それぞれ遠赤外線カメラ用のキャリブレーション画像と可視光カメラ用のキャリブレーション画像を追加します。
3. `predictor`ディレクトリに検出器(`predictor/note.txt`ファイルに記載)を追加します。

## 使用方法

1. Flaskアプリケーションを起動します:

    ```bash
    flask run
    ```

2. `http://127.0.0.1:8000/` で表示されるページからAnalysis Page `/upload` に遷移します。
3. `/upload` で指定された動画ファイルをアップロードします。
4. アップロード成功後に `/result` へ自動遷移し、サーバー側で解析が終了次第、ページ上部にグラフが表示されます。

## 参考研究

[遠赤外線画像と可視光画像を用いた口呼吸の非接触型流量推定手法の検討](https://ken.ieice.org/ken/paper/20240314dc1Q/)

電子情報通信学会技術研究報告, vol.123, no.433, MVE2023-70, pp.159-164, 2024年3月.

## 問い合わせ
[sho.akap@gmail.com](sho.akap@gmail.com)までご連絡ください。
