import pandas as pd
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance


MODE_COL_CANDIDATES = ["N1", "N2", "N3", "Qtg", "Pm", "P2", "ТРК"]
VIB_COL_CANDIDATES = [
    "Lm1",
    "F1",
    "F2",
    "2F1",
    "2F2",
    "3F1",
    "3F2",
    "Fc2",
    "Fc3",
    "Fc4",
    "Fкпа",
    "Fцс",
]


def downsample_for_plot(df, max_points=2000):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    step = max(step, 1)
    return df.iloc[::step].copy()


def load_csv_files(uploaded_files):
    frames = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, sep=";", decimal=",")
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep=";", decimal=",", encoding="cp1251")

        if "Дата и время" not in df.columns:
            st.warning(f"Файл {file.name} не содержит столбец 'Дата и время' и будет пропущен.")
            continue

        df["Дата и время"] = pd.to_datetime(
            df["Дата и время"], dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["Дата и время"])
        df["source"] = file.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("Дата и время").reset_index(drop=True)
    return data


def detect_feature_columns(df):
    mode_cols = [c for c in MODE_COL_CANDIDATES if c in df.columns]
    vib_cols = [c for c in VIB_COL_CANDIDATES if c in df.columns]
    return mode_cols, vib_cols


def train_norm_models(train_df, mode_cols, vib_cols, n_clusters=4, max_train_rows=20000):
    train_df = train_df.dropna(subset=mode_cols + vib_cols).copy()
    if len(train_df) > max_train_rows:
        step = int(np.ceil(len(train_df) / max_train_rows))
        step = max(step, 1)
        train_df = train_df.iloc[::step].copy()
    if len(train_df) < max(200, len(mode_cols) * 10):
        raise ValueError("Слишком мало строк в обучающей выборке после очистки.")

    scaler_mode = StandardScaler()
    X_mode = scaler_mode.fit_transform(train_df[mode_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_mode)
    train_df["mode_cluster"] = clusters

    models = {}
    vib_scalers = {}
    all_scores = []

    for k in sorted(np.unique(clusters)):
        subset = train_df[train_df["mode_cluster"] == k]
        if len(subset) < len(vib_cols) * 5:
            continue

        Xv = subset[vib_cols].values
        scaler_v = StandardScaler()
        Xv_scaled = scaler_v.fit_transform(Xv)

        cov = EmpiricalCovariance().fit(Xv_scaled)
        models[k] = cov
        vib_scalers[k] = scaler_v

        d2 = cov.mahalanobis(Xv_scaled)
        all_scores.extend(d2.tolist())

    if not models:
        raise ValueError("Не удалось обучить модели нормы ни для одного кластера режимов.")

    scores_arr = np.array(all_scores)
    q50 = float(np.quantile(scores_arr, 0.5))
    q995 = float(np.quantile(scores_arr, 0.995))

    model_bundle = {
        "scaler_mode": scaler_mode,
        "kmeans": kmeans,
        "models": models,
        "vib_scalers": vib_scalers,
        "q50": q50,
        "q995": q995,
        "mode_cols": mode_cols,
        "vib_cols": vib_cols,
    }
    return model_bundle


def compute_health_index(df, model_bundle, smooth_window=60):
    df = df.dropna(subset=model_bundle["mode_cols"] + model_bundle["vib_cols"]).copy()

    scaler_mode = model_bundle["scaler_mode"]
    kmeans = model_bundle["kmeans"]
    models = model_bundle["models"]
    vib_scalers = model_bundle["vib_scalers"]
    mode_cols = model_bundle["mode_cols"]
    vib_cols = model_bundle["vib_cols"]
    q50 = model_bundle["q50"]
    q995 = model_bundle["q995"]

    X_mode = scaler_mode.transform(df[mode_cols])
    df["mode_cluster"] = kmeans.predict(X_mode)

    scores = np.full(len(df), np.nan)
    for k, cov in models.items():
        mask = df["mode_cluster"] == k
        if not mask.any():
            continue
        Xv = df.loc[mask, vib_cols].values
        scaler_v = vib_scalers[k]
        Xv_scaled = scaler_v.transform(Xv)
        scores[mask] = cov.mahalanobis(Xv_scaled)

    df["anom_score"] = scores

    def normalize_score(s):
        if np.isnan(s):
            return np.nan
        return float(np.clip((s - q50) / (q995 - q50 + 1e-6), 0.0, 1.0))

    df["H"] = df["anom_score"].apply(normalize_score)
    df["H_smooth"] = df["H"].rolling(window=smooth_window, min_periods=1).median()
    return df


def main():
    st.set_page_config(page_title="Диагностика подшипников ГТД", layout="wide")
    st.title("Диагностика состояния подшипников ГТД по эксплуатационным данным")

    st.markdown(
        """
        Это приложение реализует алгоритм раннего обнаружения повреждений подшипников
        по временным рядам технологических и вибрационных параметров газотурбинных двигателей.
        Загрузите CSV-файлы с данными (`КС-1`, `КС-2`, `Обучающая выборка`), и приложение
        построит индекс технического состояния подшипников.
        """
    )

    with st.sidebar:
        st.header("Настройки алгоритма")
        n_clusters = st.slider("Число режимов работы (кластеров)", 2, 8, 4)
        smooth_window = st.slider(
            "Окно сглаживания индекса, точек", min_value=10, max_value=240, value=60, step=10
        )
        max_points = st.slider(
            "Максимальное число точек на графике",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
        )
        warn_threshold = st.slider(
            "Предупредительный порог H", min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )
        alarm_threshold = st.slider(
            "Аварийный порог H", min_value=0.0, max_value=1.0, value=0.9, step=0.05
        )

    uploaded_files = st.file_uploader(
        "Загрузите 2–3 CSV-файла с данными (разделитель ';', десятичный разделитель ',')",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Загрузите файлы для начала работы.")
        return

    data = load_csv_files(uploaded_files)
    if data.empty:
        st.error("Не удалось загрузить данные из файлов.")
        return

    st.success(f"Загружено строк: {len(data)}")

    mode_cols, vib_cols = detect_feature_columns(data)
    if len(mode_cols) < 3 or len(vib_cols) < 4:
        st.error(
            "Не найдены необходимые столбцы режимных или вибрационных параметров. "
            "Проверьте, что структура CSV соответствует ожидаемой."
        )
        st.write("Найденные режимные признаки:", mode_cols)
        st.write("Найденные вибропризнаки:", vib_cols)
        return

    st.write("Используемые режимные признаки:", mode_cols)
    st.write("Используемые вибропризнаки:", vib_cols)

    sources = sorted(data["source"].unique().tolist())
    default_train = None
    for s in sources:
        if "обуча" in s.lower():
            default_train = s
            break
    if default_train is None:
        default_train = sources[0]

    train_source = st.selectbox(
        "Файл с обучающей (эталонной) выборкой",
        options=sources,
        index=sources.index(default_train),
    )

    train_df = data[data["source"] == train_source].copy()

    st.markdown("### Обучение модели и расчёт индекса")
    st.write(
        "Нажмите кнопку ниже, чтобы обучить модель на выбранной обучающей выборке "
        "и рассчитать индекс состояния для всех загруженных данных. "
        "При изменении только порогов или интервала времени пересчёт не требуется."
    )

    run_calc = st.button("Обучить модель и рассчитать индекс", type="primary")

    data_hash = hash(
        (
            len(data),
            tuple(sorted(data["source"].unique().tolist())),
            str(data["Дата и время"].min()),
            str(data["Дата и время"].max()),
        )
    )
    current_params = {
        "train_source": train_source,
        "n_clusters": n_clusters,
        "smooth_window": smooth_window,
        "data_hash": data_hash,
    }

    need_recalc = (
        "result_df" not in st.session_state
        or "calc_params" not in st.session_state
        or st.session_state["calc_params"] != current_params
    )

    if run_calc or need_recalc:
        with st.spinner("Обучение модели нормы по обучающей выборке..."):
            try:
                model_bundle = train_norm_models(
                    train_df, mode_cols, vib_cols, n_clusters=n_clusters
                )
            except ValueError as exc:
                st.error(f"Ошибка при обучении модели: {exc}")
                return

        with st.spinner("Расчёт индекса состояния по всем данным..."):
            result_df = compute_health_index(
                data, model_bundle, smooth_window=smooth_window
            )

        st.session_state["result_df"] = result_df
        st.session_state["calc_params"] = current_params
    else:
        result_df = st.session_state.get("result_df")
        if result_df is None:
            st.info("Нажмите кнопку выше для первого запуска расчёта.")
            return

    min_ts = result_df["Дата и время"].min()
    max_ts = result_df["Дата и время"].max()

    st.subheader("График индекса технического состояния подшипников")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Интервал отображения")
        start_ts, end_ts = st.slider(
            "Диапазон времени",
            min_value=min_ts.to_pydatetime(),
            max_value=max_ts.to_pydatetime(),
            value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
            format="DD.MM.YYYY HH:mm",
        )
    with col2:
        st.write("Пороговые значения")
        st.metric("Предупредительный порог H", f"{warn_threshold:.2f}")
        st.metric("Аварийный порог H", f"{alarm_threshold:.2f}")

    mask = (result_df["Дата и время"] >= start_ts) & (
        result_df["Дата и время"] <= end_ts
    )
    view_df = result_df.loc[mask].set_index("Дата и время")

    if view_df.empty:
        st.info("На выбранном интервале нет данных для отображения.")
        return

    plot_df = view_df[["H_smooth"]].copy()
    plot_df = downsample_for_plot(plot_df, max_points=max_points)
    plot_df["Порог (предупр.)"] = warn_threshold
    plot_df["Порог (авар.)"] = alarm_threshold

    st.line_chart(plot_df)

    st.subheader("Таблица точек с превышением порогов")
    anomalies = result_df[result_df["H_smooth"] >= warn_threshold].copy()
    anomalies = anomalies.sort_values("H_smooth", ascending=False)
    if anomalies.empty:
        st.info("За выбранный интервал превышений порогового значения H не обнаружено.")
    else:
        show_cols = (
            ["Дата и время", "source", "H", "H_smooth"]
            + [c for c in vib_cols if c in anomalies.columns]
        )
        st.dataframe(anomalies[show_cols].head(200))

    st.subheader("Просмотр исходных данных")
    st.write(
        "Фрагмент исходных данных с рассчитанным индексом состояния (первые 500 строк)."
    )
    st.dataframe(result_df.head(500))


if __name__ == "__main__":
    main()

