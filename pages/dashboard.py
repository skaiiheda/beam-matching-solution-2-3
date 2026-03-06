from typing import Any, Dict

import streamlit as st

from beam_matching import (
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


def init_session_state():
    if "initialized" not in st.session_state:
        # Сначала все параметры с безопасными дефолтами
        st.session_state.twiss_in = DEFAULT_TWISS_IN
        st.session_state.twiss_target = DEFAULT_TWISS_TARGET
        st.session_state.use_penalty = False
        st.session_state.beta_limit = 10.0
        st.session_state.penalty_weight = 0.01
        st.session_state.quads = QuadrupoleSettings(k1=0.1, k2=-0.1, k3=0.1, k4=-0.1)
        st.session_state.config = DEFAULT_CONFIG

        st.session_state.initialized = True  # ← ставим ДО оптимизации

        # Оптимизация — если упадёт, параметры уже инициализированы безопасно
        result = optimize_quadrupoles(
            DEFAULT_TWISS_IN,
            DEFAULT_TWISS_TARGET,
            DEFAULT_CONFIG,
            use_penalty=False,
        )
        st.session_state.quads = result["quads"]
        st.session_state.config = BeamlineConfig(
            drift_length=result["drift_length"],
            emit_x=DEFAULT_CONFIG.emit_x,
            emit_y=DEFAULT_CONFIG.emit_y,
        )


def reset_params():
    st.session_state.twiss_in = DEFAULT_TWISS_IN
    st.session_state.twiss_target = DEFAULT_TWISS_TARGET
    st.session_state.use_penalty = False
    st.session_state.beta_limit = 10.0
    st.session_state.penalty_weight = 0.01

    # Пересчитываем оптимальные квадруполи и drift_length
    result = optimize_quadrupoles(
        DEFAULT_TWISS_IN,
        DEFAULT_TWISS_TARGET,
        DEFAULT_CONFIG,
        use_penalty=False,
    )
    st.session_state.quads = result["quads"]
    st.session_state.config = BeamlineConfig(
        drift_length=result["drift_length"],  # ← оптимальное значение
        emit_x=DEFAULT_CONFIG.emit_x,
        emit_y=DEFAULT_CONFIG.emit_y,
    )


def optimize_quads():
    """Optimize quadrupole strengths to match target Twiss."""
    with st.spinner("Оптимизация квадруполей..."):
        # Когда включён штраф за большие β, фиксируем длину дрейфа —
        # оптимизируем только квадруполи. Это предотвращает уход оптимизатора
        # в область с большим drift_length, где согласование ухудшается до 1000%+.
        optimize_drift = not st.session_state.use_penalty
        result = optimize_quadrupoles(
            st.session_state.twiss_in,
            st.session_state.twiss_target,
            st.session_state.config,
            optimize_drift=optimize_drift,
            use_penalty=st.session_state.use_penalty,
            beta_limit=st.session_state.beta_limit,
            penalty_weight=st.session_state.penalty_weight,
        )
        st.session_state.quads = result["quads"]
        st.session_state.config = BeamlineConfig(
            drift_length=result["drift_length"],
            emit_x=st.session_state.config.emit_x,
            emit_y=st.session_state.config.emit_y,
        )
        st.success(f"Оптимизировано! Финальная ошибка: {result['error']:.8e}")


def parameters_panel() -> None:
    """Create parameters input panel in sidebar."""
    st.sidebar.header("Параметры")

    # Calculate current error
    twiss_out_x, _ = propagate_through_beamline_x(
        st.session_state.twiss_in.x, st.session_state.config, st.session_state.quads
    )
    twiss_out_y, _ = propagate_through_beamline_y(
        st.session_state.twiss_in.y, st.session_state.config, st.session_state.quads
    )
    twiss_out = TwissParamsXY(x=twiss_out_x, y=twiss_out_y)

    error = calculate_matching_error(twiss_out, st.session_state.twiss_target)

    # Display error
    if error < 0.01:
        st.sidebar.metric("Ошибка согласования", f"{error:.6f}", delta_color="normal")
    elif error < 0.1:
        st.sidebar.metric("Ошибка согласования", f"{error:.6f}", delta_color="normal")
    else:
        st.sidebar.metric("Ошибка согласования", f"{error:.6f}", delta_color="inverse")

    st.sidebar.markdown("---")

    # Input Twiss Parameters
    st.sidebar.subheader("Входные параметры Твисса")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta_x = st.number_input(
            "βx (m)",
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
            "βy (m)",
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

    # Update session state
    st.session_state.twiss_in = TwissParamsXY(
        x=TwissParams(beta=beta_x, alpha=alpha_x),
        y=TwissParams(beta=beta_y, alpha=alpha_y),
    )

    st.sidebar.markdown("---")

    # Target Twiss Parameters
    st.sidebar.subheader("Целевые параметры Твисса")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta_x_t = st.number_input(
            "βx* (m)",
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
            "βy* (m)",
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

    # Update session state
    st.session_state.twiss_target = TwissParamsXY(
        x=TwissParams(beta=beta_x_t, alpha=alpha_x_t),
        y=TwissParams(beta=beta_y_t, alpha=alpha_y_t),
    )

    st.sidebar.markdown("---")

    # Beamline Configuration
    st.sidebar.subheader("Конфигурация пучкового канала")

    emit_x = st.sidebar.number_input(
        "εx (nm·rad)",
        value=st.session_state.config.emit_x * 1e9,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        key="emit_x",
    )

    emit_y = st.sidebar.number_input(
        "εy (nm·rad)",
        value=st.session_state.config.emit_y * 1e9,
        min_value=0.1,
        max_value=50.0,
        step=0.1,
        key="emit_y",
    )

    # Update session state
    st.session_state.config = BeamlineConfig(
        drift_length=st.session_state.config.drift_length,
        emit_x=emit_x * 1e-9,
        emit_y=emit_y * 1e-9,
    )

    st.sidebar.markdown("---")

    # Penalty Parameters
    st.sidebar.subheader("Параметры штрафа за большие β")

    use_penalty = st.sidebar.checkbox(
        "Использовать штраф за большие β-функции",
        value=st.session_state.use_penalty,
        key="use_penalty_checkbox",
    )

    beta_limit = st.sidebar.number_input(
        "β_limit (m)",
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

    st.session_state.use_penalty = use_penalty
    st.session_state.beta_limit = beta_limit
    st.session_state.penalty_weight = penalty_weight

    st.sidebar.markdown("---")

    # Quadrupole Strengths
    st.sidebar.subheader("Силы квадруполей (м⁻²)")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        k1 = st.number_input(
            "k₁",
            value=st.session_state.quads.k1,
            min_value=-5.0,
            max_value=5.0,
            step=0.01,
            key="k1",
        )
        k3 = st.number_input(
            "k₃",
            value=st.session_state.quads.k3,
            min_value=-5.0,
            max_value=5.0,
            step=0.01,
            key="k3",
        )
    with col2:
        k2 = st.number_input(
            "k₂",
            value=st.session_state.quads.k2,
            min_value=-5.0,
            max_value=5.0,
            step=0.01,
            key="k2",
        )
        k4 = st.number_input(
            "k₄",
            value=st.session_state.quads.k4,
            min_value=-5.0,
            max_value=5.0,
            step=0.01,
            key="k4",
        )

    # Update session state
    st.session_state.quads = QuadrupoleSettings(k1=k1, k2=k2, k3=k3, k4=k4)

    st.sidebar.markdown("---")

    # Action buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🔄 Сброс", on_click=reset_params):
            pass
    with col2:
        if st.sidebar.button("▶ Оптимизировать", on_click=optimize_quads):
            pass


def main():
    """Main dashboard page."""
    # Page config
    st.set_page_config(
        page_title="Моделирование согласования пучка",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("Dashboard")
    st.markdown("Визуализация физики ускорителя - Система 4-Quad FODO")

    # Parameters panel in sidebar
    parameters_panel()

    # Main content area
    # Row 1: Beamline diagram + Statistics table
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

        # Show matching statistics
        twiss_out_x, _ = propagate_through_beamline_x(
            st.session_state.twiss_in.x, st.session_state.config, st.session_state.quads
        )
        twiss_out_y, _ = propagate_through_beamline_y(
            st.session_state.twiss_in.y, st.session_state.config, st.session_state.quads
        )
        twiss_out = TwissParamsXY(x=twiss_out_x, y=twiss_out_y)

        stats = calculate_matching_statistics(twiss_out, st.session_state.twiss_target)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if stats["is_matched"]:
                st.success(f"✓ Согласовано: Средняя ошибка {stats['avg_error']:.2f}%")
            else:
                st.warning(
                    f"✗ Не согласовано: Средняя ошибка {stats['avg_error']:.2f}%"
                )
        with col_s2:
            st.metric("Макс. ошибка", f"{stats['max_error']:.2f}%")

    # Row 2: Beta plot
    st.markdown("---")
    st.subheader("β-функции вдоль пучкового канала")
    fig_beta = create_beta_plot(
        st.session_state.twiss_in,
        st.session_state.config,
        st.session_state.quads,
        st.session_state.twiss_target,
    )
    st.plotly_chart(fig_beta, use_container_width=True)

    # Row 3: Phase space + Envelope
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Фазовое пространство")
        tab_x, tab_y = st.tabs(["Плоскость X (x, x')", "Плоскость Y (y, y')"])

        fig_phase_x, fig_phase_y = create_phase_space_plot(
            st.session_state.twiss_in,
            st.session_state.config,
            st.session_state.quads,
            st.session_state.twiss_target,
        )

        with tab_x:
            st.plotly_chart(fig_phase_x, use_container_width=True)

        with tab_y:
            st.plotly_chart(fig_phase_y, use_container_width=True)

        st.caption(
            "Эллипсы представляют контур эмиттанса пучка: γx² + 2αxx' + βx'² = ε"
        )

    with col2:
        st.subheader("Огибающая пучка")
        fig_envelope = create_envelope_plot(
            st.session_state.twiss_in, st.session_state.config, st.session_state.quads
        )
        st.plotly_chart(fig_envelope, use_container_width=True)

        # Show emittance info
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.metric(
                "Эмиттанс X", f"{st.session_state.config.emit_x * 1e9:.1f} нм·рад"
            )
        with col_e2:
            st.metric(
                "Эмиттанс Y", f"{st.session_state.config.emit_y * 1e9:.1f} нм·рад"
            )

    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.caption(
            "Матрицы переноса: Дрейф [[1, L], [0, 1]] | Квадруполь [[1, 0], [-k, 1]]"
        )
    with col_f2:
        st.caption("Распространение Твисса: β₂ = m₁₁²β₁ - 2m₁₁m₁₂α₁ + m₁₂²γ₁")
    with col_f3:
        st.caption("Огибающая пучка: σ = √(β·ε), где ε - эмиттанс")


if __name__ == "__main__":
    main()
