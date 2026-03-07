from typing import Any, Dict

import streamlit as st

from beam_matching.optics import (
    DEFAULT_CONFIG,
    DEFAULT_TWISS_IN,
    DEFAULT_TWISS_TARGET,
    BeamlineConfig,
    QuadrupoleSettings,
    TwissParams,
    TwissParamsXY,
    calculate_matching_error,
    optimize_quadrupoles,
    propagate_through_beamline_x,
    propagate_through_beamline_y,
)
from components import (
    calculate_matching_statistics,
    create_beamline_diagram,
    create_beta_plot,
    create_envelope_plot,
    create_phase_space_plot,
    create_statistics_table,
)

# ---------------------------------------------------------------------------
# Кэш пропагации — ключ: хэш всех параметров
# ---------------------------------------------------------------------------


@st.cache_data(max_entries=128, show_spinner=False)
def _cached_propagate(
    # TwissParams разбиты на скаляры — st.cache_data требует хэшируемые аргументы
    bx_in: float,
    ax_in: float,
    by_in: float,
    ay_in: float,
    drift: float,
    ql: float,
    n_quads: int,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,
    pts: int = 50,
):
    """
    Кэшированная пропагация Твисса через весь канал.
    Возвращает (twiss_out_x, history_x, twiss_out_y, history_y).
    При любом изменении параметров Streamlit автоматически инвалидирует кэш.
    """
    twiss_in_x = TwissParams(beta=bx_in, alpha=ax_in)
    twiss_in_y = TwissParams(beta=by_in, alpha=ay_in)
    config = BeamlineConfig(
        drift_length=drift,
        emit_x=1.0,
        emit_y=1.0,
        quad_length=ql,
        n_quads=n_quads,
    )
    quads = QuadrupoleSettings(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)
    rx, hx = propagate_through_beamline_x(twiss_in_x, config, quads, pts)
    ry, hy = propagate_through_beamline_y(twiss_in_y, config, quads, pts)
    return rx, hx, ry, hy


def _propagate_current() -> tuple:
    """Пропагирует текущие параметры из session_state с кэшем."""
    s = st.session_state
    q = s.quads
    c = s.config
    ti = s.twiss_in
    rx, hx, ry, hy = _cached_propagate(
        ti.x.beta,
        ti.x.alpha,
        ti.y.beta,
        ti.y.alpha,
        c.drift_length,
        c.quad_length,
        c.n_quads,
        q.k1,
        q.k2,
        q.k3,
        q.k4,
        q.k5,
    )
    return TwissParamsXY(x=rx, y=ry), hx, hy


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------


def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.twiss_in = DEFAULT_TWISS_IN
        st.session_state.twiss_target = DEFAULT_TWISS_TARGET
        st.session_state.use_penalty = True
        st.session_state.beta_limit = 6.0
        st.session_state.penalty_weight = 0.1
        st.session_state.n_quads = 4
        st.session_state.quads = QuadrupoleSettings(k1=0.1, k2=-0.1, k3=0.1, k4=-0.1)
        st.session_state.config = DEFAULT_CONFIG

        st.session_state.initialized = True

        # Первичная оптимизация
        result = optimize_quadrupoles(
            DEFAULT_TWISS_IN,
            DEFAULT_TWISS_TARGET,
            DEFAULT_CONFIG,
            use_penalty=True,
            beta_limit=6.0,
            penalty_weight=0.1,
        )
        st.session_state.quads = result["quads"]
        st.session_state.config = BeamlineConfig(
            drift_length=result["drift_length"],
            emit_x=DEFAULT_CONFIG.emit_x,
            emit_y=DEFAULT_CONFIG.emit_y,
            quad_length=DEFAULT_CONFIG.quad_length,
            n_quads=4,
        )


def reset_params():
    n = st.session_state.n_quads
    cfg = BeamlineConfig(
        drift_length=1.0,
        emit_x=DEFAULT_CONFIG.emit_x,
        emit_y=DEFAULT_CONFIG.emit_y,
        quad_length=DEFAULT_CONFIG.quad_length,
        n_quads=n,
    )
    st.session_state.twiss_in = DEFAULT_TWISS_IN
    st.session_state.twiss_target = DEFAULT_TWISS_TARGET
    st.session_state.use_penalty = True
    st.session_state.beta_limit = 6.0
    st.session_state.penalty_weight = 0.1

    result = optimize_quadrupoles(
        DEFAULT_TWISS_IN,
        DEFAULT_TWISS_TARGET,
        cfg,
        use_penalty=True,
        beta_limit=6.0,
        penalty_weight=0.1,
    )
    st.session_state.quads = result["quads"]
    st.session_state.config = BeamlineConfig(
        drift_length=result["drift_length"],
        emit_x=DEFAULT_CONFIG.emit_x,
        emit_y=DEFAULT_CONFIG.emit_y,
        quad_length=DEFAULT_CONFIG.quad_length,
        n_quads=n,
    )


def optimize_quads():
    with st.spinner("Оптимизация квадруполей..."):
        result = optimize_quadrupoles(
            st.session_state.twiss_in,
            st.session_state.twiss_target,
            st.session_state.config,
            use_penalty=st.session_state.use_penalty,
            beta_limit=st.session_state.beta_limit,
            penalty_weight=st.session_state.penalty_weight,
        )
        st.session_state.quads = result["quads"]
        st.success(f"Оптимизировано! Финальная ошибка: {result['error']:.8e}")


# ---------------------------------------------------------------------------
# Боковая панель параметров
# ---------------------------------------------------------------------------


def parameters_panel() -> None:
    st.sidebar.header("Параметры")

    # --- Ошибка согласования ---
    twiss_out, _, _ = _propagate_current()
    error = calculate_matching_error(twiss_out, st.session_state.twiss_target)
    st.sidebar.metric(
        "Ошибка согласования",
        f"{error:.6f}",
        delta_color="normal" if error < 0.1 else "inverse",
    )
    st.sidebar.markdown("---")

    # --- Число квадруполей ---
    st.sidebar.subheader("Конфигурация пучкового канала")

    n_quads_new = st.sidebar.radio(
        "Число квадруполей",
        options=[4, 5],
        index=0 if st.session_state.n_quads == 4 else 1,
        horizontal=True,
        key="n_quads_radio",
        help="4 — стандартная FODO-ячейка; 5 — дополнительная степень свободы для сложного согласования",
    )

    # При смене числа квадруполей обновляем конфиг и сбрасываем k5
    if n_quads_new != st.session_state.n_quads:
        st.session_state.n_quads = n_quads_new
        st.session_state.config = BeamlineConfig(
            drift_length=st.session_state.config.drift_length,
            emit_x=st.session_state.config.emit_x,
            emit_y=st.session_state.config.emit_y,
            quad_length=st.session_state.config.quad_length,
            n_quads=n_quads_new,
        )
        # k5 обнуляем при переходе на 4
        if n_quads_new == 4:
            q = st.session_state.quads
            st.session_state.quads = QuadrupoleSettings(
                k1=q.k1, k2=q.k2, k3=q.k3, k4=q.k4, k5=0.0
            )

    emit_x = st.sidebar.number_input(
        "εx (нм·рад)",
        value=st.session_state.config.emit_x * 1e9,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        key="emit_x",
    )
    emit_y = st.sidebar.number_input(
        "εy (нм·рад)",
        value=st.session_state.config.emit_y * 1e9,
        min_value=0.1,
        max_value=50.0,
        step=0.1,
        key="emit_y",
    )
    quad_length = st.sidebar.number_input(
        "Длина квадруполя (м)",
        value=st.session_state.config.quad_length,
        min_value=0.02,
        max_value=0.5,
        step=0.01,
        format="%.2f",
        key="quad_length",
        help="Физическая длина каждого квадруполя. Тонкая линза = 0, реалистично 0.05–0.2 м",
    )

    st.session_state.config = BeamlineConfig(
        drift_length=st.session_state.config.drift_length,
        emit_x=emit_x * 1e-9,
        emit_y=emit_y * 1e-9,
        quad_length=quad_length,
        n_quads=st.session_state.n_quads,
    )
    st.sidebar.markdown("---")

    # --- Входные параметры Твисса ---
    st.sidebar.subheader("Входные параметры Твисса")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta_x = st.number_input(
            "βx (м)",
            value=st.session_state.twiss_in.x.beta,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            key="beta_x_in",
        )
    with col2:
        alpha_x = st.number_input(
            "αx",
            value=st.session_state.twiss_in.x.alpha,
            min_value=-10.0,
            max_value=10.0,
            step=0.1,
            key="alpha_x_in",
        )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta_y = st.number_input(
            "βy (м)",
            value=st.session_state.twiss_in.y.beta,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            key="beta_y_in",
        )
    with col2:
        alpha_y = st.number_input(
            "αy",
            value=st.session_state.twiss_in.y.alpha,
            min_value=-10.0,
            max_value=10.0,
            step=0.1,
            key="alpha_y_in",
        )
    st.session_state.twiss_in = TwissParamsXY(
        x=TwissParams(beta=beta_x, alpha=alpha_x),
        y=TwissParams(beta=beta_y, alpha=alpha_y),
    )
    st.sidebar.markdown("---")

    # --- Целевые параметры ---
    st.sidebar.subheader("Целевые параметры Твисса")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta_x_t = st.number_input(
            "βx* (м)",
            value=st.session_state.twiss_target.x.beta,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            key="beta_x_target",
        )
    with col2:
        alpha_x_t = st.number_input(
            "αx*",
            value=st.session_state.twiss_target.x.alpha,
            min_value=-10.0,
            max_value=10.0,
            step=0.1,
            key="alpha_x_target",
        )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta_y_t = st.number_input(
            "βy* (м)",
            value=st.session_state.twiss_target.y.beta,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            key="beta_y_target",
        )
    with col2:
        alpha_y_t = st.number_input(
            "αy*",
            value=st.session_state.twiss_target.y.alpha,
            min_value=-10.0,
            max_value=10.0,
            step=0.1,
            key="alpha_y_target",
        )
    st.session_state.twiss_target = TwissParamsXY(
        x=TwissParams(beta=beta_x_t, alpha=alpha_x_t),
        y=TwissParams(beta=beta_y_t, alpha=alpha_y_t),
    )
    st.sidebar.markdown("---")

    # --- Штраф за большие β ---
    st.sidebar.subheader("Параметры штрафа за большие β")
    st.sidebar.checkbox(
        "Использовать штраф за большие β",
        value=st.session_state.use_penalty,
        key="use_penalty",
    )
    beta_limit = st.sidebar.number_input(
        "β_limit (м)",
        value=st.session_state.beta_limit,
        min_value=1.0,
        max_value=50.0,
        step=0.1,
        key="beta_limit_input",
    )
    penalty_weight = st.sidebar.number_input(
        "Вес штрафа",
        value=st.session_state.penalty_weight,
        min_value=0.001,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="penalty_weight_input",
    )
    st.session_state.beta_limit = beta_limit
    st.session_state.penalty_weight = penalty_weight
    st.sidebar.markdown("---")

    # --- Силы квадруполей ---
    n = st.session_state.n_quads
    st.sidebar.subheader(f"Силы квадруполей (м⁻²) — {n} шт.")

    q = st.session_state.quads

    if n == 4:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            k1 = st.number_input(
                "k₁", value=q.k1, min_value=-10.0, max_value=10.0, step=0.01, key="k1"
            )
            k3 = st.number_input(
                "k₃", value=q.k3, min_value=-10.0, max_value=10.0, step=0.01, key="k3"
            )
        with col2:
            k2 = st.number_input(
                "k₂", value=q.k2, min_value=-10.0, max_value=10.0, step=0.01, key="k2"
            )
            k4 = st.number_input(
                "k₄", value=q.k4, min_value=-10.0, max_value=10.0, step=0.01, key="k4"
            )
        k5 = 0.0
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            k1 = st.number_input(
                "k₁", value=q.k1, min_value=-10.0, max_value=10.0, step=0.01, key="k1"
            )
            k3 = st.number_input(
                "k₃", value=q.k3, min_value=-10.0, max_value=10.0, step=0.01, key="k3"
            )
            k5 = st.number_input(
                "k₅",
                value=q.k5,
                min_value=-10.0,
                max_value=10.0,
                step=0.01,
                key="k5",
                help="Дополнительный квадруполь Q5 — доступен только в режиме 5 линз",
            )
        with col2:
            k2 = st.number_input(
                "k₂", value=q.k2, min_value=-10.0, max_value=10.0, step=0.01, key="k2"
            )
            k4 = st.number_input(
                "k₄", value=q.k4, min_value=-10.0, max_value=10.0, step=0.01, key="k4"
            )

    st.session_state.quads = QuadrupoleSettings(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)
    st.sidebar.markdown("---")

    # --- Кнопки ---
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🔄 Сброс", on_click=reset_params):
            pass
    with col2:
        if st.sidebar.button("▶ Оптимизировать", on_click=optimize_quads):
            pass


# ---------------------------------------------------------------------------
# Главная страница
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Моделирование согласования пучка",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    n = st.session_state.n_quads
    st.title("Dashboard")
    st.markdown(f"Визуализация физики ускорителя — Система **4/5-Quad**")

    parameters_panel()

    # --- Строка 1: схема + статистика ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Компоновка пучкового канала")
        fig_beamline = create_beamline_diagram(
            st.session_state.quads, st.session_state.config
        )
        st.plotly_chart(fig_beamline, use_container_width=True)

    with col2:
        st.subheader("Статистика согласования")
        df_stats = create_statistics_table(
            st.session_state.twiss_in,
            st.session_state.config,
            st.session_state.quads,
            st.session_state.twiss_target,
        )
        st.dataframe(df_stats, use_container_width=True, height=300)

        twiss_out, _, _ = _propagate_current()
        stats = calculate_matching_statistics(twiss_out, st.session_state.twiss_target)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if stats["is_matched"]:
                st.success(f"✓ Согласовано: Ср. ошибка {stats['avg_error']:.2f}%")
            else:
                st.warning(f"✗ Не согласовано: Ср. ошибка {stats['avg_error']:.2f}%")
        with col_s2:
            st.metric("Макс. ошибка", f"{stats['max_error']:.2f}%")

    # --- Строка 2: β-функции ---
    st.markdown("---")
    st.subheader("β-функции вдоль пучкового канала")
    fig_beta = create_beta_plot(
        st.session_state.twiss_in,
        st.session_state.config,
        st.session_state.quads,
        st.session_state.twiss_target,
    )
    st.plotly_chart(fig_beta, use_container_width=True)

    # --- Строка 3: фазовое пространство + огибающая ---
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Фазовое пространство")

        # Опции позиций динамически зависят от n_quads
        position_options = (
            ["Вход (начало)"]
            + [f"После Q{i}" for i in range(1, n + 1)]
            + ["Выход (конец)"]
        )
        selected_idx = st.selectbox(
            "Выберите позицию:",
            range(len(position_options)),
            format_func=lambda i: position_options[i],
            key="phase_space_position",
        )

        tab_x, tab_y = st.tabs(["Плоскость X (x, x')", "Плоскость Y (y, y')"])
        fig_phase_x, fig_phase_y = create_phase_space_plot(
            st.session_state.twiss_in,
            st.session_state.config,
            st.session_state.quads,
            st.session_state.twiss_target,
            selected_position_index=selected_idx,
        )
        with tab_x:
            st.plotly_chart(fig_phase_x, use_container_width=True)
        with tab_y:
            st.plotly_chart(fig_phase_y, use_container_width=True)
        st.caption("Эллипсы: γx² + 2αxx' + βx'² = ε")

    with col2:
        st.subheader("Огибающая пучка")
        fig_envelope = create_envelope_plot(
            st.session_state.twiss_in, st.session_state.config, st.session_state.quads
        )
        st.plotly_chart(fig_envelope, use_container_width=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.metric(
                "Эмиттанс X", f"{st.session_state.config.emit_x * 1e9:.1f} нм·рад"
            )
        with col_e2:
            st.metric(
                "Эмиттанс Y", f"{st.session_state.config.emit_y * 1e9:.1f} нм·рад"
            )

    # --- Подвал ---
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.caption("Матрицы переноса: Дрейф [[1, L], [0, 1]] | Квадруполь (толстый)")
    with col_f2:
        st.caption("Распространение Твисса: β₂ = m₁₁²β₁ - 2m₁₁m₁₂α₁ + m₁₂²γ₁")
    with col_f3:
        st.caption("Огибающая пучка: σ = √(β·ε)")


if __name__ == "__main__":
    main()
