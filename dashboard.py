import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from supabase import create_client, Client

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
# CLIENTES DE BASES DE DATOS (CON CACHÉ)
# ==============================================================================
#@st.cache_resource
#def get_supabase_client() -> Client:
#    url = os.getenv("SUPABASE_URL")
#    key = os.getenv("SUPABASE_KEY")
#    if not url or not key:
#        st.error("Credenciales de Supabase no configuradas en el archivo .env")
#    return create_client(url, key)

#@st.cache_resource
#def get_qdrant_client() -> QdrantClient:
#    host = os.getenv("QDRANT_HOST", "localhost")
#    port = int(os.getenv("QDRANT_PORT", "6333"))
#    return QdrantClient(host=host, port=port)

@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    # 1. Intentar leer de st.secrets (para Streamlit Cloud) o de os.getenv (para local/.env)
    qdrant_url = st.secrets.get("QDRANT_URL") if hasattr(st, "secrets") and "QDRANT_URL" in st.secrets else os.getenv("QDRANT_URL")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY") if hasattr(st, "secrets") and "QDRANT_API_KEY" in st.secrets else os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Fallback local
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") and "SUPABASE_URL" in st.secrets else os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") and "SUPABASE_KEY" in st.secrets else os.getenv("SUPABASE_KEY")
    return create_client(url, key)

# ==============================================================================
# FUNCIONES DE CONSULTA Y PERSISTENCIA
# ==============================================================================
def fetch_universe():
    supabase = get_supabase_client()
    try:
        res = supabase.table("asset_universe").select("*").order("tipo_instrumento").order("codigo").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error consultando 'asset_universe': {e}")
        return pd.DataFrame()


def fetch_scout_dates():
    supabase = get_supabase_client()
    try:
        res = supabase.table("market_scout_rankings").select("fecha_proceso").order("fecha_proceso", desc=True).execute()
        if not res.data:
            return []
        dates = sorted(list({r["fecha_proceso"] for r in res.data}), reverse=True)
        return dates
    except Exception:
        return []


def fetch_scout_rankings(fecha: str):
    supabase = get_supabase_client()
    try:
        res = supabase.table("market_scout_rankings").select("*").eq("fecha_proceso", fecha).execute()
        if not res.data:
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        # Conversión de columnas numéricas
        num_cols = ["precio_actual", "rendimiento_1w", "rendimiento_1m", "rendimiento_6m", "rendimiento_1y", "dividend_yield", "volumen_promedio"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        st.error(f"Error consultando 'market_scout_rankings': {e}")
        return pd.DataFrame()


def fetch_trades():
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


def fetch_memories():
    qc = get_qdrant_client()
    try:
        collections = [c.name for c in qc.get_collections().collections]
        if "market_lessons" not in collections:
            return []
        points, _ = qc.scroll(collection_name="market_lessons", limit=100, with_payload=True)
        return [{"id": p.id, **(p.payload or {})} for p in points]
    except Exception:
        return []


# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================
st.sidebar.title("🎛️ Centro de Mando")
if st.sidebar.button("🔄 Refrescar Tablero", width='stretch'):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado del Sistema:**")
st.sidebar.write(f"• **Supabase DB:** {'🟢 Conectado' if os.getenv('SUPABASE_URL') else '🔴 Desconectado'}")
st.sidebar.write(f"• **Qdrant Vector DB:** {'🟢 Conectado' if os.getenv('QDRANT_HOST') else '🔴 Desconectado'}")
st.sidebar.write(f"• **Broker API:** Alpaca Paper Cloud")
st.sidebar.write(f"• **Motor LLM:** Claude 3.5 Sonnet")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Puedes modificar los activos activos en la pestaña 'Administrar Universo' para afinar el escaneo de mañana.")

# ==============================================================================
# PANEL PRINCIPAL Y PESTAÑAS
# ==============================================================================
st.title("📊 Terminal Cuantitativa y Célula Multi-Agente")
st.caption("Supervisión del universo analizable, rankings de Yahoo Finance, bitácora de ejecución y memoria heurística.")

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
        # Filtros de encabezado
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

            # Aplicar filtros
            filtered_scout = df_scout.copy()
            if selected_tipo != "TODOS":
                filtered_scout = filtered_scout[filtered_scout["tipo_instrumento"] == selected_tipo]
            if selected_rec != "TODAS":
                filtered_scout = filtered_scout[filtered_scout["recomendacion"] == selected_rec]

            # Ordenar de mayor a menor
            col_to_sort = sort_metric[1]
            filtered_scout = filtered_scout.sort_values(by=col_to_sort, ascending=False).reset_index(drop=True)

            # Tarjetas de resumen métrico rápido
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

            # Gráfico de barras Plotly de los Top 10 según orden
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
                st.plotly_chart(fig_scout, width='stretch')

            # Tabla interactiva con formato adaptativo por columnas
            st.markdown("#### 📋 Matriz Detallada de Rendimientos y Ratios")

            # Configuración de columnas de Streamlit
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
                width='stretch',
                hide_index=True
            )

            # Visualizador de descripciones de empresa / fondo / contrato
            st.markdown("#### 🔍 Inspeccionar Perfil Corporativo o de Contrato")
            selected_inspect = st.selectbox(
                "Selecciona un activo para ver su descripción de negocio completa:",
                options=filtered_scout["codigo"].tolist()
            )
            if selected_inspect:
                row_sel = filtered_scout[filtered_scout["codigo"] == selected_inspect].iloc[0]
                c_desc1, c_desc2 = st.columns([3, 1])
                with c_desc1:
                    st.info(f"**Descripción del Activo ({row_sel['nombre']}):**\n\n{row_sel.get('descripcion', 'Sin descripción.')}")
                with c_desc2:
                    st.write("**Metadatos Adicionales:**")
                    st.json(row_sel.get("metrica_extra", {}))
        else:
            st.info("Sin registros para la fecha seleccionada.")


# ==============================================================================
# PESTAÑA 2: ADMINISTRAR UNIVERSO DE ACTIVOS (ASSET_UNIVERSE)
# ==============================================================================
with tab_universe:
    st.subheader("⚙️ Control del Universo de Activos (`asset_universe`)")
    st.write(
        "Administra qué instrumentos serán analizados por `market_scout.py`. "
        "Los activos marcados como **Pausados** se omitirán del escaneo diario para ahorrar tiempo y ancho de banda."
    )

    supabase = get_supabase_client()
    df_univ = fetch_universe()

    # Formulario para Agregar Activo
    with st.expander("➕ Agregar Nuevo Activo al Universo", expanded=False):
        with st.form("form_add_asset", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns([2, 3, 2])
            with f_col1:
                in_code = st.text_input("Código / Ticker bursátil *", placeholder="ej. PLTR, ETH-USD, EURUSD=X").strip().upper()
            with f_col2:
                in_name = st.text_input("Nombre de la Empresa o Instrumento", placeholder="ej. Palantir Technologies Inc.")
            with f_col3:
                in_type = st.selectbox("Tipo de Instrumento *", ["ACCIONES", "DIVIDENDOS", "FONDOS", "CRIPTOS", "FOREX", "OPCIONES"])

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
                        st.success(f"Activo {in_code} añadido exitosamente como activo habilitado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al agregar activo: {e}")

    # Controles para Pausar / Reactivar y Remover
    st.markdown("#### 🔄 Modificar Estado o Eliminar Activo Existente")
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
            if st.button(toggle_label, width='stretch'):
                try:
                    supabase.table("asset_universe").update({"activo": not current_status}).eq("codigo", selected_ticker).execute()
                    st.toast(f"{selected_ticker} ahora está {'Activo' if not current_status else 'Pausado'}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error actualizando estado: {e}")

        with col_act3:
            st.write("Acción definitiva:")
            if st.button("🗑️ Eliminar del Universo", type="secondary", width='stretch'):
                try:
                    supabase.table("asset_universe").delete().eq("codigo", selected_ticker).execute()
                    st.toast(f"Activo {selected_ticker} eliminado permanentemente.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al eliminar: {e}")

        with col_act4:
            st.write("Filtro por tipo de lista:")
            filter_univ_type = st.selectbox("Ver:", ["TODOS"] + sorted(list(df_univ["tipo_instrumento"].unique())))

        # Resumen cuantitativo del universo
        u_m1, u_m2, u_m3, u_m4 = st.columns(4)
        u_m1.metric("Total en Catálogo", len(df_univ))
        u_m2.metric("Habilitados (Activos)", len(df_univ[df_univ["activo"] == True]))
        u_m3.metric("Pausados", len(df_univ[df_univ["activo"] == False]))
        u_m4.metric("Clases de Activo", df_univ["tipo_instrumento"].nunique())

        # Tabla de visualización del universo
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
            width='stretch',
            hide_index=True
        )
    else:
        st.info("No hay activos registrados en 'asset_universe'. Ejecuta `python sync_universe.py` o agrega uno arriba.")


# ==============================================================================
# PESTAÑA 3: BITÁCORA TRANSACCIONAL (TRADE_JOURNAL)
# ==============================================================================
with tab_trades:
    st.subheader("📋 Registro de Órdenes Ejecutadas por la Célula")
    df_trades = fetch_trades()

    if df_trades.empty:
        st.info("Sin operaciones registradas aún. Corre `python main.py` para generar trades.")
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
                "quantity": st.column_config.NumberColumn("Acciones", format="%.0f")
            },
            width='stretch',
            hide_index=True
        )

        st.markdown("#### 🧠 Inspección de Rationale Operativo y Auditoría")
        selected_trade_id = st.selectbox("Selecciona ID de Trade:", filtered_trades["id"].tolist())
        if selected_trade_id:
            row_tr = filtered_trades[filtered_trades["id"] == selected_trade_id].iloc[0]
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.markdown("**Tesis / Rationale del CIO:**")
                st.info(row_tr.get("rationale", "Sin registro"))
            with c_r2:
                st.markdown("**Evaluación Post-Trade del Auditor:**")
                st.success(row_tr.get("post_analysis", "Sin registro"))


# ==============================================================================
# PESTAÑA 4: TELEMETRÍA Y MÉTRICAS DE CARTERA
# ==============================================================================
with tab_analytics:
    st.subheader("📈 Distribución de Capital y Exposición")
    df_trades = fetch_trades()

    if not df_trades.empty:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_pie = px.pie(
                df_trades,
                names="symbol",
                values="total_value",
                title="Distribución de Capital por Activo",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, width='stretch')

        with col_g2:
            fig_bar = px.bar(
                df_trades,
                x="created_at",
                y="total_value",
                color="action",
                title="Historial de Operaciones por Timestamp",
                labels={"total_value": "Monto Total (USD)", "created_at": "Fecha/Hora"}
            )
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, width='stretch')
    else:
        st.info("Sin datos suficientes en la bitácora para generar gráficos de cartera.")


# ==============================================================================
# PESTAÑA 5: MEMORIA VECTORIAL EPISÓDICA (QDRANT)
# ==============================================================================
with tab_memory:
    st.subheader("🧠 Colección Vectorial `market_lessons` (Qdrant)")
    st.write("Lecciones cuantitativas, heurísticas y advertencias de riesgo indexadas por los agentes.")

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
