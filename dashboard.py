import os
import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import yfinance as yf

from qdrant_client import QdrantClient
from supabase import create_client, Client

from alpaca.trading.client import TradingClient

# ==============================================================================
# CONFIGURACIÓN GENERAL Y ESTILOS
# ==============================================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(
    page_title="Trading Desk & Market Scout | Telemetría",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    .badge-buy { background-color: #0e6251; color: #a3e4d7; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-hold { background-color: #7d6608; color: #f9e79f; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-sell { background-color: #78281f; color: #f5b7b1; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# RESUELVE SECRETOS (STREAMLIT CLOUD / LOCAL .ENV)
# ==============================================================================
def get_config_val(key: str, default=None):
    """Obtiene credenciales desde st.secrets o variables de entorno."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# ==============================================================================
# CLIENTES DE BASES DE DATOS Y BROKER (CON CACHÉ)
# ==============================================================================
@st.cache_resource
def get_supabase_client() -> Client:
    url = get_config_val("SUPABASE_URL")
    key = get_config_val("SUPABASE_KEY")
    if not url or not key:
        st.error("Credenciales de Supabase no configuradas.")
    return create_client(url, key)


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    qdrant_url = get_config_val("QDRANT_URL")
    qdrant_api_key = get_config_val("QDRANT_API_KEY")

    if qdrant_url:
        url_clean = str(qdrant_url).strip()
        if not url_clean.startswith("http://") and not url_clean.startswith("https://"):
            url_clean = f"https://{url_clean}"
        return QdrantClient(url=url_clean, api_key=qdrant_api_key if qdrant_api_key else None)



@st.cache_resource
def get_alpaca_client() -> TradingClient | None:
    api_key = get_config_val("ALPACA_API_KEY")
    secret_key = get_config_val("ALPACA_SECRET_KEY")
    paper_str = str(get_config_val("ALPACA_PAPER", "True")).lower()
    is_paper = paper_str in ("true", "1")

    if not api_key or not secret_key or "XXX" in api_key:
        return None
    try:
        return TradingClient(api_key, secret_key, paper=is_paper)
    except Exception as e:
        st.error(f"Error conectando a Alpaca: {e}")
        return None


# ==============================================================================
# CONSULTAS A SUPABASE, QDRANT Y ALPACA
# ==============================================================================
def fetch_universe() -> pd.DataFrame:
    supabase = get_supabase_client()
    try:
        res = supabase.table("asset_universe").select("*").order("tipo_instrumento").order("codigo").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error consultando 'asset_universe': {e}")
        return pd.DataFrame()


def fetch_scout_dates() -> List[str]:
    supabase = get_supabase_client()
    try:
        res = supabase.table("market_scout_rankings").select("fecha_proceso").order("fecha_proceso", desc=True).execute()
        if not res.data:
            return []
        return sorted(list({r["fecha_proceso"] for r in res.data}), reverse=True)
    except Exception:
        return []


def fetch_scout_rankings(fecha: str) -> pd.DataFrame:
    supabase = get_supabase_client()
    try:
        res = supabase.table("market_scout_rankings").select("*").eq("fecha_proceso", fecha).execute()
        if not res.data:
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        num_cols = ["precio_actual", "rendimiento_1w", "rendimiento_1m", "rendimiento_6m", "rendimiento_1y", "dividend_yield", "volumen_promedio"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        st.error(f"Error consultando 'market_scout_rankings': {e}")
        return pd.DataFrame()


def fetch_trades() -> pd.DataFrame:
    supabase = get_supabase_client()
    try:
        res = supabase.table("trade_journal").select("*").order("created_at", desc=True).execute()
        if not res.data:
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["total_value"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0) * pd.to_numeric(df["price"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        st.error(f"Error consultando 'trade_journal': {e}")
        return pd.DataFrame()


def fetch_memories() -> List[dict]:
    qc = get_qdrant_client()
    try:
        collections = [c.name for c in qc.get_collections().collections]
        if "market_lessons" not in collections:
            return []
        points, _ = qc.scroll(collection_name="market_lessons", limit=100, with_payload=True)
        return [{"id": p.id, **(p.payload or {})} for p in points]
    except Exception:
        return []


def fetch_alpaca_portfolio_data() -> Tuple[dict, pd.DataFrame]:
    """Obtiene métricas de cuenta y posiciones activas desde Alpaca con su clasificación."""
    client = get_alpaca_client()
    if not client:
        return {}, pd.DataFrame()

    try:
        account = client.get_account()
        positions_raw = client.get_all_positions()

        account_info = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "initial_margin": float(account.initial_margin)
        }

        if not positions_raw:
            return account_info, pd.DataFrame()

        rows = []
        df_univ = fetch_universe()
        univ_map = {}
        if not df_univ.empty and "codigo" in df_univ.columns:
            univ_map = dict(zip(df_univ["codigo"], df_univ["tipo_instrumento"]))

        for p in positions_raw:
            raw_sym = p.symbol.upper()
            # Mapear formato cripto Alpaca (BTC/USD -> BTC-USD) para consulta
            clean_sym = raw_sym.replace("/", "-")
            
            # Clasificar tipo de instrumento
            inst_type = univ_map.get(clean_sym) or univ_map.get(raw_sym)
            if not inst_type:
                if "-USD" in clean_sym or "/USD" in raw_sym:
                    inst_type = "CRIPTOS"
                elif raw_sym in ["SPY", "QQQ", "IWM", "VOO", "VTI", "SMH", "XLK", "XLF"]:
                    inst_type = "FONDOS"
                else:
                    inst_type = "ACCIONES"

            mv = float(p.market_value)
            rows.append({
                "symbol": clean_sym,
                "symbol_alpaca": raw_sym,
                "qty": float(p.qty),
                "market_value": mv,
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "tipo_instrumento": inst_type
            })

        df_positions = pd.DataFrame(rows)
        return account_info, df_positions
    except Exception as e:
        st.error(f"Error consultando portafolio de Alpaca: {e}")
        return {}, pd.DataFrame()


# ==============================================================================
# MOTOR DE HISTORIAL PONDERADO Y BENCHMARKS
# ==============================================================================
@st.cache_data(ttl=300)
def fetch_historical_series(symbols: List[str], period: str) -> pd.DataFrame:
    """Descarga cotizaciones históricas para activos y benchmarks desde Yahoo Finance."""
    interval_map = {
        "5d": "15m",
        "1mo": "1d",
        "6mo": "1d",
        "1y": "1d",
        "5y": "1wk"
    }
    interval = interval_map.get(period, "1d")
    
    # Benchmarks de mercado solicitados
    benchmarks = ["^NYA", "^GSPC", "^DJI", "^MXX"]
    all_tickers = list(dict.fromkeys(symbols + benchmarks))

    try:
        df = yf.download(
            tickers=all_tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )
    except Exception:
        return pd.DataFrame()

    closes = pd.DataFrame()
    for t in all_tickers:
        try:
            if len(all_tickers) == 1:
                closes[t] = df["Close"]
            else:
                if t in df and "Close" in df[t]:
                    closes[t] = df[t]["Close"]
        except Exception:
            continue

    closes = closes.ffill().bfill()
    return closes


# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================
st.sidebar.title("🎛️ Centro de Mando")
if st.sidebar.button("🔄 Refrescar Tablero", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado del Sistema:**")
st.sidebar.write(f"• **Supabase DB:** {'🟢 Conectado' if get_config_val('SUPABASE_URL') else '🔴 Desconectado'}")
st.sidebar.write(f"• **Qdrant Vector DB:** {'🟢 Conectado' if get_config_val('QDRANT_URL') else '🔴 Desconectado'}")
st.sidebar.write(f"• **Broker API:** {'🟢 Alpaca Conectado' if get_config_val('ALPACA_API_KEY') else '🔴 Alpaca Desconectado'}")
st.sidebar.write(f"• **Motor LLM:** Claude 3.5 Sonnet")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Revisa la pestaña 'Métricas y Cartera' para auditar la exposición viva de Alpaca y contrastarla contra el NYSE, S&P 500, Dow Jones y el IPC Mexicano.")

# ==============================================================================
# PANEL PRINCIPAL Y PESTAÑAS
# ==============================================================================
st.title("📊 Terminal Cuantitativa y Célula Multi-Agente")
st.caption("Supervisión de cartera en vivo en Alpaca, simulación teórica contra índices, rankings de Yahoo Finance y memoria heurística.")

tab_scout, tab_universe, tab_trades, tab_analytics, tab_memory = st.tabs([
    "🏆 Explorador de Rankings (Top 30)",
    "⚙️ Administrar Universo de Activos",
    "📋 Bitácora Transaccional",
    "📈 Métricas y Cartera",
    "🧠 Memoria Vectorial (Qdrant)"
])


# ==============================================================================
# PESTAÑA 1: EXPLORADOR VISUAL DE RANKINGS
# ==============================================================================
with tab_scout:
    st.subheader("🎯 Oportunidades Filtradas por Yahoo Finance (Top 30)")
    available_dates = fetch_scout_dates()

    if not available_dates:
        st.warning("No se encontraron escaneos en 'market_scout_rankings'. Ejecuta primero `python market_scout.py`.")
    else:
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 3])
        with f_col1:
            selected_date = st.selectbox("📅 Fecha de Proceso:", available_dates, index=0)

        df_scout = fetch_scout_rankings(selected_date)

        if not df_scout.empty:
            with f_col2:
                tipo_options = ["TODOS"] + sorted(list(df_scout["tipo_instrumento"].unique()))
                selected_tipo = st.selectbox("🏷️ Tipo de Instrumento:", tipo_options)

            with f_col3:
                rec_options = ["TODAS"] + sorted(list(df_scout["recomendacion"].dropna().unique()))
                selected_rec = st.selectbox("⭐ Recomendación:", rec_options)

            with f_col4:
                sort_metric = st.selectbox(
                    "📊 Ordenar por Mayor Rendimiento / Métrica:",
                    options=[
                        ("Rendimiento Último Año (1Y)", "rendimiento_1y"),
                        ("Rendimiento Semestral (6M)", "rendimiento_6m"),
                        ("Rendimiento Mensual (1M)", "rendimiento_1m"),
                        ("Rendimiento Semanal (1W)", "rendimiento_1w"),
                        ("Dividend Yield (%)", "dividend_yield"),
                        ("Precio Actual", "precio_actual"),
                        ("Volumen Transaccional", "volumen_promedio")
                    ],
                    format_func=lambda x: x[0]
                )

            filtered_scout = df_scout.copy()
            if selected_tipo != "TODOS":
                filtered_scout = filtered_scout[filtered_scout["tipo_instrumento"] == selected_tipo]
            if selected_rec != "TODAS":
                filtered_scout = filtered_scout[filtered_scout["recomendacion"] == selected_rec]

            col_to_sort = sort_metric[1]
            filtered_scout = filtered_scout.sort_values(by=col_to_sort, ascending=False).reset_index(drop=True)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Activos Listados", len(filtered_scout))
            max_asset = filtered_scout.iloc[0]["codigo"] if not filtered_scout.empty else "N/A"
            max_val = filtered_scout.iloc[0][col_to_sort] if not filtered_scout.empty else 0.0

            suffix = "%" if "rendimiento" in col_to_sort or "dividend" in col_to_sort else (" USD" if "precio" in col_to_sort else "")
            k2.metric(f"Líder en {sort_metric[0][:15]}...", f"{max_asset}", f"{max_val:+.2f}{suffix}")

            avg_1y = filtered_scout["rendimiento_1y"].mean() if not filtered_scout.empty else 0.0
            k3.metric("Rendimiento 1Y Promedio", f"{avg_1y:+.2f}%")

            buy_count = (filtered_scout["recomendacion"].str.contains("BUY", na=False)).sum()
            k4.metric("Con Voto 'BUY' / 'STRONG BUY'", f"{buy_count} activos")

            st.markdown("---")

            if not filtered_scout.empty:
                top_10_chart = filtered_scout.head(10).copy()
                fig_scout = px.bar(
                    top_10_chart,
                    x="codigo",
                    y=col_to_sort,
                    color=col_to_sort,
                    text=col_to_sort,
                    title=f"Top 10 Activos Ordenados por: {sort_metric[0]}",
                    color_continuous_scale="Viridis",
                    labels={"codigo": "Activo", col_to_sort: sort_metric[0]}
                )
                fig_scout.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_scout.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_scout, use_container_width=True)

            st.markdown("#### 📋 Matriz Detallada de Rendimientos y Ratios")
            column_configuration = {
                "codigo": st.column_config.TextColumn("Código / Ticker", width="small"),
                "nombre": st.column_config.TextColumn("Nombre del Activo", width="medium"),
                "tipo_instrumento": st.column_config.TextColumn("Tipo", width="small"),
                "categoria": st.column_config.TextColumn("Categoría / Sector", width="medium"),
                "recomendacion": st.column_config.TextColumn("Consenso Yahoo", width="small"),
                "precio_actual": st.column_config.NumberColumn("Precio Spot", format="$%.2f"),
                "rendimiento_1w": st.column_config.NumberColumn("1 Sem (1W)", format="%+.2f%%"),
                "rendimiento_1m": st.column_config.NumberColumn("1 Mes (1M)", format="%+.2f%%"),
                "rendimiento_6m": st.column_config.NumberColumn("6 Meses (6M)", format="%+.2f%%"),
                "rendimiento_1y": st.column_config.NumberColumn("1 Año (1Y)", format="%+.2f%%"),
                "dividend_yield": st.column_config.NumberColumn("Div. Yield", format="%.2f%%"),
                "volumen_promedio": st.column_config.NumberColumn("Volumen Prom.", format="%,.0f")
            }

            display_cols = [
                "codigo", "nombre", "tipo_instrumento", "categoria", "recomendacion",
                "precio_actual", "rendimiento_1w", "rendimiento_1m", "rendimiento_6m",
                "rendimiento_1y", "dividend_yield", "volumen_promedio"
            ]

            st.dataframe(
                filtered_scout[display_cols],
                column_config=column_configuration,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Sin registros para la fecha seleccionada.")


# ==============================================================================
# PESTAÑA 2: ADMINISTRAR UNIVERSO DE ACTIVOS (ASSET_UNIVERSE)
# ==============================================================================
with tab_universe:
    st.subheader("⚙️ Control del Universo de Activos (`asset_universe`)")
    supabase = get_supabase_client()
    df_univ = fetch_universe()

    with st.expander("➕ Agregar Nuevo Activo al Universo", expanded=False):
        with st.form("form_add_asset", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns([2, 3, 2])
            with f_col1:
                in_code = st.text_input("Código / Ticker bursátil *", placeholder="ej. PLTR, BTC-USD, SPY").strip().upper()
            with f_col2:
                in_name = st.text_input("Nombre de la Empresa o Instrumento", placeholder="ej. Palantir Technologies Inc.")
            with f_col3:
                in_type = st.selectbox("Tipo de Instrumento *", ["ACCIONES", "DIVIDENDOS", "FONDOS", "BONOS", "CRIPTOS", "FOREX", "OPCIONES"])

            submit_add = st.form_submit_button("Guardar en el Universo")
            if submit_add:
                if not in_code:
                    st.error("El código del ticker es obligatorio.")
                else:
                    try:
                        record_add = {
                            "codigo": in_code,
                            "nombre": in_name if in_name else in_code,
                            "tipo_instrumento": in_type,
                            "activo": True,
                            "origen": "MANUAL",
                            "actualizado_el": pd.Timestamp.utcnow().isoformat()
                        }
                        supabase.table("asset_universe").upsert(record_add, on_conflict="codigo").execute()
                        st.success(f"Activo {in_code} añadido exitosamente.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al agregar activo: {e}")

    if not df_univ.empty:
        col_act1, col_act2, col_act3, col_act4 = st.columns([3, 2, 2, 2])
        with col_act1:
            selected_ticker = st.selectbox(
                "Selecciona el activo a gestionar:",
                options=df_univ["codigo"].tolist(),
                format_func=lambda x: f"{x} - {df_univ.loc[df_univ['codigo'] == x, 'nombre'].values[0]} ({df_univ.loc[df_univ['codigo'] == x, 'tipo_instrumento'].values[0]})"
            )

        asset_row = df_univ[df_univ["codigo"] == selected_ticker].iloc[0]
        current_status = bool(asset_row.get("activo", True))

        with col_act2:
            st.write(f"Estado actual: **{'🟢 Activo' if current_status else '⏸️ Pausado'}**")
            toggle_label = "Pausar Activo" if current_status else "Reactivar Activo"
            if st.button(toggle_label, use_container_width=True):
                try:
                    supabase.table("asset_universe").update({"activo": not current_status}).eq("codigo", selected_ticker).execute()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error actualizando estado: {e}")

        with col_act3:
            st.write("Acción definitiva:")
            if st.button("🗑️ Eliminar del Universo", type="secondary", use_container_width=True):
                try:
                    supabase.table("asset_universe").delete().eq("codigo", selected_ticker).execute()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al eliminar: {e}")

        with col_act4:
            st.write("Filtro por tipo de lista:")
            filter_univ_type = st.selectbox("Ver:", ["TODOS"] + sorted(list(df_univ["tipo_instrumento"].unique())))

        view_univ = df_univ.copy()
        if filter_univ_type != "TODOS":
            view_univ = view_univ[view_univ["tipo_instrumento"] == filter_univ_type]

        st.dataframe(
            view_univ[["codigo", "nombre", "tipo_instrumento", "activo", "origen", "actualizado_el"]],
            column_config={
                "codigo": st.column_config.TextColumn("Código", width="small"),
                "nombre": st.column_config.TextColumn("Nombre", width="medium"),
                "tipo_instrumento": st.column_config.TextColumn("Categoría", width="small"),
                "activo": st.column_config.CheckboxColumn("Estado Habilitado", width="small"),
                "origen": st.column_config.TextColumn("Origen", width="small"),
                "actualizado_el": st.column_config.DatetimeColumn("Última Sincro", format="YYYY-MM-DD HH:mm")
            },
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# PESTAÑA 3: BITÁCORA TRANSACCIONAL (TRADE_JOURNAL)
# ==============================================================================
with tab_trades:
    st.subheader("📋 Registro de Órdenes Ejecutadas por la Célula")
    df_trades = fetch_trades()

    if df_trades.empty:
        st.info("Sin operaciones registradas aún en Supabase (`trade_journal`).")
    else:
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            sym_filter = st.multiselect("Filtrar por Ticker:", options=df_trades["symbol"].unique(), default=df_trades["symbol"].unique())
        with t_col2:
            act_filter = st.multiselect("Filtrar por Acción:", options=df_trades["action"].unique(), default=df_trades["action"].unique())

        filtered_trades = df_trades[
            (df_trades["symbol"].isin(sym_filter)) &
            (df_trades["action"].isin(act_filter))
        ]

        st.dataframe(
            filtered_trades[["id", "created_at", "symbol", "action", "quantity", "price", "total_value", "status", "order_id"]],
            column_config={
                "price": st.column_config.NumberColumn("Precio", format="$%.2f"),
                "total_value": st.column_config.NumberColumn("Volumen Total", format="$%.2f"),
                "quantity": st.column_config.NumberColumn("Cantidad", format="%.2f")
            },
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# PESTAÑA 4: TELEMETRÍA, CARTERA EN ALPACA Y SIMULACIÓN HISTÓRICA
# ==============================================================================
with tab_analytics:
    st.subheader("📈 Cartera Activa en Alpaca y Simulación Histórica Teórica")

    acct_info, df_pos = fetch_alpaca_portfolio_data()

    if not acct_info:
        st.warning("⚠️ No se pudo obtener la información de Alpaca. Verifica tus credenciales `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`.")
    else:
        # 1. KPIs del Portafolio en Vivo
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Valor Total (Equity)", f"${acct_info['equity']:,.2f}")
        
        cash_val = acct_info['cash']
        kpi2.metric(
            "Efectivo Líquido",
            f"${cash_val:,.2f}",
            delta="Margen en uso" if cash_val < 0 else "Liquidez positiva",
            delta_color="inverse" if cash_val < 0 else "normal"
        )
        
        kpi3.metric("Poder de Compra (BP)", f"${acct_info['buying_power']:,.2f}")
        
        total_mv = df_pos["market_value"].sum() if not df_pos.empty else 0.0
        kpi4.metric("Capital Invertido", f"${total_mv:,.2f}")

        total_upl = df_pos["unrealized_pl"].sum() if not df_pos.empty else 0.0
        kpi5.metric("PnL No Realizado", f"${total_upl:+,.2f}", delta=f"{total_upl:+,.2f}")

        st.markdown("---")

        if df_pos.empty:
            st.info("La cuenta de Alpaca no tiene posiciones abiertas en este momento.")
        else:
            # 2. FILTRO POR TIPO DE INSTRUMENTO
            f_col1, f_col2 = st.columns([2, 4])
            with f_col1:
                available_types = ["TODOS"] + sorted(list(df_pos["tipo_instrumento"].unique()))
                selected_inst_filter = st.selectbox("🎯 Filtrar Cartera por Tipo de Instrumento:", available_types)

            df_pos_filtered = df_pos.copy()
            if selected_inst_filter != "TODOS":
                df_pos_filtered = df_pos_filtered[df_pos_filtered["tipo_instrumento"] == selected_inst_filter]

            # 3. GRÁFICAS DE PAY (PIE CHARTS) DE COMPOSICIÓN
            st.markdown("#### 🥧 Distribución del Portafolio Activo")
            c_pie1, c_pie2 = st.columns(2)

            with c_pie1:
                # Composición por Activo Individual
                fig_pie_assets = px.pie(
                    df_pos_filtered,
                    names="symbol",
                    values="market_value",
                    title=f"Composición por Activo Individual ({selected_inst_filter})",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_pie_assets.update_traces(textinfo="percent+label", pull=[0.02] * len(df_pos_filtered))
                fig_pie_assets.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_pie_assets, use_container_width=True)

            with c_pie2:
                # Composición por Clase de Activo (con o sin Efectivo)
                pie_class_data = df_pos.groupby("tipo_instrumento")["market_value"].sum().reset_index()
                if cash_val > 0:
                    pie_class_data = pd.concat([
                        pie_class_data,
                        pd.DataFrame([{"tipo_instrumento": "EFECTIVO (CASH)", "market_value": cash_val}])
                    ], ignore_index=True)

                fig_pie_class = px.pie(
                    pie_class_data,
                    names="tipo_instrumento",
                    values="market_value",
                    title="Distribución Global por Clase de Activo y Liquidez",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_pie_class.update_traces(textinfo="percent+label")
                fig_pie_class.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_pie_class, use_container_width=True)

            # 4. TABLA DETALLADA DE POSICIONES
            with st.expander("🔎 Ver Tabla Detallada de Posiciones en Alpaca", expanded=False):
                st.dataframe(
                    df_pos_filtered[[
                        "symbol", "tipo_instrumento", "qty", "current_price",
                        "market_value", "avg_entry_price", "unrealized_pl", "unrealized_plpc"
                    ]],
                    column_config={
                        "symbol": st.column_config.TextColumn("Ticker"),
                        "tipo_instrumento": st.column_config.TextColumn("Tipo"),
                        "qty": st.column_config.NumberColumn("Cantidad", format="%.4f"),
                        "current_price": st.column_config.NumberColumn("Precio Spot", format="$%.2f"),
                        "market_value": st.column_config.NumberColumn("Valor de Mercado", format="$%.2f"),
                        "avg_entry_price": st.column_config.NumberColumn("Precio Promedio", format="$%.2f"),
                        "unrealized_pl": st.column_config.NumberColumn("PnL ($)", format="$%+.2f"),
                        "unrealized_plpc": st.column_config.NumberColumn("PnL (%)", format="%+.2f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")

            # 5. SIMULACIÓN HISTÓRICA TEÓRICA vs BENCHMARKS INTERNACIONALES
            st.markdown("#### 📊 Comportamiento Histórico Teórico vs Índices de Referencia")
            st.caption(
                "Simula el retorno porcentual que habría obtenido la combinación actual de tu portafolio en comparación directa con el "
                "**NYSE Composite (^NYA)**, **S&P 500 (^GSPC)**, **Dow Jones (^DJI)** y el **IPC Mexicano (^MXX)**."
            )

            # Controles de temporalidad
            t_col, _ = st.columns([3, 3])
            with t_col:
                timeframe_label = st.radio(
                    "Selecciona el horizonte temporal:",
                    options=["1W (1 Semana)", "1M (1 Mes)", "6M (6 Meses)", "1Y (1 Año)", "5Y (5 Años)"],
                    index=3,
                    horizontal=True
                )

            period_key_map = {
                "1W (1 Semana)": "5d",
                "1M (1 Mes)": "1mo",
                "6M (6 Meses)": "6mo",
                "1Y (1 Año)": "1y",
                "5Y (5 Años)": "5y"
            }
            selected_period = period_key_map[timeframe_label]

            # Activos a simular (según el filtro seleccionado)
            symbols_to_simulate = df_pos_filtered["symbol"].tolist()

            if not symbols_to_simulate:
                st.warning(f"No hay activos en la cartera para la categoría: {selected_inst_filter}.")
            else:
                with st.spinner("Descargando series históricas y computando curvas normalizadas..."):
                    hist_data = fetch_historical_series(symbols_to_simulate, selected_period)

                if hist_data.empty:
                    st.error("No se pudieron recuperar las cotizaciones históricas desde Yahoo Finance.")
                else:
                    # Normalizar curvas a rendimiento acumulado base 0%
                    pct_df = pd.DataFrame(index=hist_data.index)

                    # Ponderación del portafolio actual
                    total_alloc_value = df_pos_filtered["market_value"].sum()
                    weights = {
                        row["symbol"]: (row["market_value"] / total_alloc_value)
                        for _, row in df_pos_filtered.iterrows()
                    }

                    # Curva ponderada del portafolio
                    port_returns = pd.Series(0.0, index=hist_data.index)
                    valid_weights_sum = 0.0

                    for sym in symbols_to_simulate:
                        if sym in hist_data and not hist_data[sym].dropna().empty:
                            s = hist_data[sym].dropna()
                            base_price = s.iloc[0]
                            if base_price > 0:
                                sym_return = ((s / base_price) - 1.0) * 100.0
                                w = weights.get(sym, 0.0)
                                port_returns = port_returns.add(sym_return * w, fill_value=0.0)
                                valid_weights_sum += w

                    # Reescalar si algún símbolo no tuvo datos completos
                    if valid_weights_sum > 0:
                        port_returns = port_returns / valid_weights_sum

                    pct_df["Portafolio Teórico"] = port_returns

                    # Curvas de Benchmarks
                    benchmark_labels = {
                        "^NYA": "NYSE Composite (^NYA)",
                        "^GSPC": "S&P 500 (^GSPC)",
                        "^DJI": "Dow Jones 30 (^DJI)",
                        "^MXX": "S&P/BMV IPC México (^MXX)"
                    }

                    for b_ticker, b_name in benchmark_labels.items():
                        if b_ticker in hist_data and not hist_data[b_ticker].dropna().empty:
                            b_series = hist_data[b_ticker].dropna()
                            b_base = b_series.iloc[0]
                            if b_base > 0:
                                pct_df[b_name] = ((b_series / b_base) - 1.0) * 100.0

                    # Graficar con Plotly
                    fig_comp = go.Figure()

                    # Línea del Portafolio (resaltada en color verde)
                    fig_comp.add_trace(go.Scatter(
                        x=pct_df.index,
                        y=pct_df["Portafolio Teórico"],
                        mode="lines",
                        name=f"🚀 Portafolio Teórico ({selected_inst_filter})",
                        line=dict(color="#00FFAA", width=3.5)
                    ))

                    # Líneas de Benchmarks
                    colors_bench = {
                        "NYSE Composite (^NYA)": "#3498DB",
                        "S&P 500 (^GSPC)": "#F39C12",
                        "Dow Jones 30 (^DJI)": "#9B59B6",
                        "S&P/BMV IPC México (^MXX)": "#E74C3C"
                    }

                    for b_name, b_col in colors_bench.items():
                        if b_name in pct_df.columns:
                            fig_comp.add_trace(go.Scatter(
                                x=pct_df.index,
                                y=pct_df[b_name],
                                mode="lines",
                                name=b_name,
                                line=dict(color=b_col, width=1.8, dash="dot" if "IPC" in b_name else "solid")
                            ))

                    fig_comp.update_layout(
                        template="plotly_dark",
                        height=520,
                        title=f"Rendimiento Porcentual Acumulado — Período: {timeframe_label} (Filtro: {selected_inst_filter})",
                        xaxis_title="Fecha",
                        yaxis_title="Retorno Acumulado (%)",
                        yaxis=dict(ticksuffix="%"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=20, r=20, t=70, b=20),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_comp, use_container_width=True)

                    # Tabla comparativa de retornos acumulados finales
                    st.markdown("##### 📌 Comparativa de Retornos en el Período Seleccionado")
                    m_cols = st.columns(len(pct_df.columns))
                    for idx, col_name in enumerate(pct_df.columns):
                        val_end = pct_df[col_name].iloc[-1] if not pct_df.empty else 0.0
                        m_cols[idx].metric(col_name.replace(" (^NYA)", "").replace(" (^GSPC)", "").replace(" (^DJI)", "").replace(" (^MXX)", ""), f"{val_end:+.2f}%")


# ==============================================================================
# PESTAÑA 5: MEMORIA VECTORIAL EPISÓDICA (QDRANT)
# ==============================================================================
with tab_memory:
    st.subheader("🧠 Colección Vectorial `market_lessons` (Qdrant)")
    memories = fetch_memories()

    if not memories:
        st.info("No se registran vectores en la colección 'market_lessons' de Qdrant.")
    else:
        q_filter = st.text_input("Filtrar lecciones por palabra clave (ej. dividend, stop, riesgo):", "")
        filtered_mems = [
            m for m in memories
            if q_filter.lower() in str(m.get("rationale", "")).lower() or
               q_filter.lower() in str(m.get("post_analysis", "")).lower() or
               q_filter.lower() in str(m.get("symbol", "")).lower()
        ] if q_filter else memories

        st.caption(f"Exponiendo {len(filtered_mems)} lecciones.")
        for mem in filtered_mems:
            with st.expander(f"📍 Punto #{mem.get('id', 'N/A')} — Ticker: {mem.get('symbol', 'N/A')}"):
                st.markdown(f"**Tesis Registrada:**\n> {mem.get('rationale', 'N/A')}")
                st.markdown(f"**Directriz Post-Operación:**\n> {mem.get('post_analysis', 'N/A')}")
