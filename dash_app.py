import os
# Fix for macOS fork safety crash
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, DiskcacheManager
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import os
import sys
import tempfile
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from openai import OpenAI
import diskcache

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from tandon_ai_doc_intel import DocumentPipeline

# --- Cache Setup for Background Callbacks ---
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP], 
                suppress_callback_exceptions=True, 
                background_callback_manager=background_callback_manager)
app.title = "Tandon AI Doc Intel"

import logging
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="pypdf")

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_ai_insight(context_data, analysis_type, api_key):
    if not api_key:
        return "⚠️ OpenAI API Key required for insights."
    
    client = OpenAI(api_key=api_key)
    prompts = {
        "pipeline": f"Analyze these pipeline latencies (seconds): {context_data}. Identify bottlenecks.",
        "risk": f"Review this risk distribution: {context_data}. Summarize the overall risk profile of the corpus.",
        "cluster": f"Interpret these document clusters found via PCA/K-Means: {context_data}. Suggest potential thematic groupings."
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a senior data analyst. Be concise and professional."},
                {"role": "user", "content": prompts.get(analysis_type, str(context_data))}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

def parse_contents(contents, filename, api_key):
    # API key is optional now
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    logger.info(f"🚀 Starting pipeline for: {filename}")
    pipeline = DocumentPipeline(openai_api_key=api_key)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(decoded)
        tmp_path = tmp.name
    
    try:
        result = pipeline.process(tmp_path)
        result.filename = filename
        logger.info(f"✅ Successfully processed: {filename}")
        return result, None
    except Exception as e:
        logger.error(f"❌ Failed to process {filename}: {str(e)}", exc_info=True)
        return None, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Layout ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H2(["📄 Tandon AI Document Intelligence"], className="text-center my-4 text-light"), width=12)
    ]),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚙️ Configuration", className="text-light"),
                dbc.CardBody([
                    dbc.Label("OpenAI API Key", className="text-light"),
                    dbc.Input(id="api-key", placeholder="sk-...", type="password", className="mb-3 text-light bg-dark", style={"border": "1px solid #6c757d"}),
                    
                    dbc.Label("Upload Documents", className="text-light"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="bi bi-cloud-upload me-2"), 
                            "Drag & Drop PDFs"
                        ], className="text-light"),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0',
                            'borderColor': '#adb5bd', 'cursor': 'pointer'
                        },
                        multiple=True
                    ),
                    html.Div(id='upload-status', className="mt-2 text-info small"),
                    
                    html.Hr(className="text-light"),
                    dbc.Button("Run Analysis", id="btn-run", color="primary", className="w-100", disabled=True),
                    
                    # Progress Bar (Initially Hidden)
                    html.Div(id="progress-wrapper", children=[
                        html.Div(className="d-flex justify-content-between text-light small mt-2", children=[
                            html.Span("Processing..."),
                            html.Span(id="progress-label", children="0%")
                        ]),
                        dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, style={"height": "10px"}, className="mt-1"),
                    ], style={"display": "none"}, className="mt-3"),
                    
                    dbc.Spinner(html.Div(id="loading-output"), color="light", size="sm", spinner_class_name="mt-2")
                ])
            ], className="mb-4 bg-dark border-secondary"),
            
            # Global Metrics Card
            dbc.Card([
                dbc.CardHeader("📊 Global KPIs", className="text-light"),
                dbc.CardBody(id="global-kpis", children="Run analysis to see metrics.", className="text-light")
            ], className="bg-dark border-secondary")
            
        ], width=3),
        
        # Main Content
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📈 Corpus Dashboard", tab_id="tab-1", label_style={"color": "#adb5bd"}, active_label_style={"color": "#fff", "fontWeight": "bold"}),
                dbc.Tab(label="🤖 ML & Clusters", tab_id="tab-2", label_style={"color": "#adb5bd"}, active_label_style={"color": "#fff", "fontWeight": "bold"}),
                dbc.Tab(label="🔍 Individual Inspector", tab_id="tab-3", label_style={"color": "#adb5bd"}, active_label_style={"color": "#fff", "fontWeight": "bold"}),
            ], id="tabs", active_tab="tab-1", className="mb-3"),
            
            html.Div(id="tab-content", className="text-light")
        ], width=9)
    ]),
    
    dcc.Store(id='stored-data')
    
], fluid=True, className="p-4 bg-black", style={"min-height": "100vh"})

# --- Callbacks ---

# 1. Update Upload Status (Immediate Feedback)
@callback(
    Output('upload-status', 'children'),
    Output('btn-run', 'disabled'),
    Input('upload-data', 'filename')
)
def update_upload_status(filenames):
    if not filenames:
        return "No files selected.", True
    return f"✅ {len(filenames)} files ready to process.", False

# 2. Run Pipeline (Heavy Lifting)
@callback(
    Output('stored-data', 'data'),
    Output('loading-output', 'children'),
    Input('btn-run', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('api-key', 'value'),
    background=True,
    running=[
        (Output("btn-run", "disabled"), True, False),
        (Output("progress-wrapper", "style"), {"display": "block"}, {"display": "none"}),
    ],
    progress=[
        Output("progress-bar", "value"), 
        Output("progress-label", "children")
    ],
    prevent_initial_call=True
)
def process_documents(set_progress, n_clicks, list_of_contents, list_of_names, api_key):
    if not list_of_contents:
        return no_update, "No files."
    
    results_data = []
    total = len(list_of_contents)
    
    for i, (c, n) in enumerate(zip(list_of_contents, list_of_names)):
        # Update progress
        percent = int(((i) / total) * 100)
        set_progress((str(percent), f"{percent}%"))
        
        logger.info(f"Processing file {i+1}/{total}: {n}")
        res, err = parse_contents(c, n, api_key)
        if err:
            logger.error(f"Skipping {n} due to error: {err}")
            continue # Skip errors for batch
        
        # Serialize
        results_data.append({
            "filename": res.filename,
            "timings": res.processing_time_seconds, # Full timing dict
            "readability": res.readability_score,
            "gunning_fog": res.gunning_fog,
            "risk": res.risk_analysis.get("risk_level", "Unknown") if res.risk_analysis else "Unknown",
            "risk_factors": res.risk_analysis.get("risk_factors", []) if res.risk_analysis else [],
            "type": "Digital" if res.metadata.get("is_digital_pdf") else "Scanned",
            "embeddings": res.embeddings,
            "summary": res.summary,
            "sentiment": res.sentiment_polarity,
            "subjectivity": res.sentiment_subjectivity,
            "lexical_diversity": res.lexical_diversity,
            "topics": res.topics
        })
        
        # Update progress after step
        percent = int(((i + 1) / total) * 100)
        set_progress((str(percent), f"{percent}%"))
    
    if not results_data and total > 0:
         return [], "⚠️ All files failed. Check API Key or console logs."

    return results_data, "" # Clear loading text when done

# 3. Update Global KPIs (Sidebar)
@callback(
    Output('global-kpis', 'children'),
    Input('stored-data', 'data')
)
def update_kpis(data):
    if not data:
        return "Run analysis to see metrics."
    
    df = pd.DataFrame(data)
    avg_latency = df['timings'].apply(lambda x: x.get('Total', 0)).mean()
    avg_readability = df['readability'].mean()
    high_risk = len(df[df['risk'] == 'High'])
    
    return html.Div([
        html.H6(f"Docs Processed: {len(df)}"),
        html.Hr(className="my-2"),
        html.P(f"⏱ Avg Latency: {avg_latency:.2f}s"),
        html.P(f"📖 Avg Readability: {avg_readability:.1f}"),
        html.P(f"⚠️ High Risk Docs: {high_risk}", className="text-danger" if high_risk > 0 else "text-success")
    ])

# 4. Render Tabs
@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("stored-data", "data"),
    State("api-key", "value")
)
def render_tabs(active_tab, data, api_key):
    if not data:
        return html.Div("Please upload documents and click 'Run Analysis'.", className="text-center mt-5 text-muted")
    
    df = pd.DataFrame(data)
    
    if active_tab == "tab-1":
        # --- DASHBOARD ---
        
        # Risk Pie Chart
        fig_risk = px.pie(df, names="risk", title="Risk Distribution", template="plotly_dark", 
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Readability vs Latency Scatter
        df['latency_total'] = df['timings'].apply(lambda x: x.get('Total', 0))
        fig_scatter = px.scatter(df, x="readability", y="latency_total", color="type", hover_data=["filename"], 
                                 title="Complexity vs. Performance", template="plotly_dark")
        
        # Pipeline Stage Breakdown (Avg)
        # Extract timings for all docs and average them
        stage_keys = ["Ingestion", "Classification", "Extraction", "Enrichment", "Validation", "Analytics", "Embedding"]
        avg_timings = {k: 0 for k in stage_keys}
        for item in data:
            t = item.get('timings', {})
            for k in stage_keys:
                avg_timings[k] += t.get(k, 0)
        
        # Average it
        for k in avg_timings:
            avg_timings[k] /= len(data)
            
        fig_pipeline = px.bar(x=list(avg_timings.keys()), y=list(avg_timings.values()), 
                              title="Avg Pipeline Latency by Stage (s)", template="plotly_dark",
                              labels={'x':'Stage', 'y':'Seconds'})

        # AI Insight for Risk
        risk_counts = df['risk'].value_counts().to_dict()
        ai_risk_msg = get_ai_insight(risk_counts, "risk", api_key)

        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("🤖 AI Risk Assessment"), dbc.CardBody(ai_risk_msg)], className="bg-dark border-info mb-3"), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_risk), width=4),
                dbc.Col(dcc.Graph(figure=fig_scatter), width=4),
                dbc.Col(dcc.Graph(figure=fig_pipeline), width=4)
            ])
        ])
    
    elif active_tab == "tab-2":
        # --- ML & CLUSTERS ---
        
        embeddings = [d['embeddings'] for d in data if d.get('embeddings')]
        cluster_graph = html.Div("Need at least 3 documents for clustering.", className="text-warning")
        
        if len(embeddings) >= 3:
            X = np.array(embeddings)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            kmeans = KMeans(n_clusters=min(3, len(embeddings)), random_state=42)
            clusters = kmeans.fit_predict(X)
            
            df_pca = pd.DataFrame({
                "x": components[:, 0], "y": components[:, 1],
                "Cluster": [f"Group {c+1}" for c in clusters],
                "Filename": [d['filename'] for d in data if d.get('embeddings')]
            })
            fig_cluster = px.scatter(df_pca, x="x", y="y", color="Cluster", hover_data=["Filename"], 
                                     title="Semantic Clusters (PCA Reduced)", template="plotly_dark")
            
            # AI Insight for Clusters
            cluster_summary = {f"Group {c+1}": int(count) for c, count in zip(*np.unique(clusters, return_counts=True))}
            ai_cluster_msg = get_ai_insight(cluster_summary, "cluster", api_key)
            
            cluster_graph = html.Div([
                dbc.Card([dbc.CardHeader("🤖 AI Cluster Analysis"), dbc.CardBody(ai_cluster_msg)], className="bg-dark border-info mb-3"),
                dcc.Graph(figure=fig_cluster)
            ])

        return html.Div([
            dbc.Row([
                dbc.Col(cluster_graph, width=8),
                dbc.Col([
                    html.H5("ML Metrics Reference", className="text-light"),
                    html.Ul([
                        html.Li(f"Avg Sentiment: {df['sentiment'].mean():.2f} (-1 to 1)", className="text-light"),
                        html.Li(f"Avg Subjectivity: {df['subjectivity'].mean():.2f} (0 to 1)", className="text-light"),
                        html.Li(f"Avg Gunning Fog: {df['gunning_fog'].mean():.1f} (Grade Level)", className="text-light"),
                        html.Li(f"Avg Lexical Diversity: {df['lexical_diversity'].mean():.2f}", className="text-light")
                    ], className="text-light")
                ], width=4)
            ])
        ])
    
    elif active_tab == "tab-3":
        # --- INSPECTOR ---
        
        options = [{"label": d['filename'], "value": i} for i, d in enumerate(data)]
        
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="doc-selector", 
                    options=options, 
                    value=0, 
                    clearable=False,
                    style={'color': 'black'} 
                ), width=6, className="mb-3")
            ]),
            html.Div(id="inspector-content")
        ])

# 5. Inspector Callback (Sub-callback for Tab 3)
@callback(
    Output("inspector-content", "children"),
    Input("doc-selector", "value"),
    State("stored-data", "data")
)
def update_inspector(selected_idx, data):
    if selected_idx is None or not data:
        return ""
    
    doc = data[int(selected_idx)]
    
    # Timeline Chart for this specific doc
    timings = doc.get('timings', {})
    # Remove 'Total' for chart
    clean_timings = {k:v for k,v in timings.items() if k != 'Total'}
    fig_time = px.bar(x=list(clean_timings.keys()), y=list(clean_timings.values()), 
                      title="Pipeline Execution Time (s)", template="plotly_dark",
                      labels={'x':'Stage', 'y':'Seconds'})
    fig_time.update_layout(autosize=False, height=400) # Fix height expansion issue
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📄 Document Summary", className="text-light"),
                    dbc.CardBody(doc['summary'], style={"max-height": "200px", "overflow-y": "auto"}, className="text-light")
                ], className="mb-3 bg-dark border-secondary"),
                
                dbc.Card([
                    dbc.CardHeader("⚠️ Risk Factors", className="text-light"),
                    dbc.CardBody([html.Li(r) for r in doc['risk_factors']] if doc['risk_factors'] else "No specific risks identified.", className="text-light")
                ], className="mb-3 bg-dark border-warning" if doc['risk'] == 'High' else "mb-3 bg-dark border-success")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Metrics", className="text-light"),
                    dbc.ListGroup([
                        dbc.ListGroupItem(f"Readability: {doc['readability']}", className="bg-dark text-light"),
                        dbc.ListGroupItem(f"Gunning Fog: {doc['gunning_fog']}", className="bg-dark text-light"),
                        dbc.ListGroupItem(f"Sentiment: {doc['sentiment']:.2f}", className="bg-dark text-light"),
                        dbc.ListGroupItem(f"Diversity: {doc['lexical_diversity']:.2f}", className="bg-dark text-light"),
                    ], flush=True)
                ], className="mb-3 bg-dark border-secondary"),
                
                dcc.Graph(figure=fig_time)
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5("🏷️ Extracted Topics", className="text-light"),
                html.P(", ".join(doc.get('topics', [])) if doc.get('topics') else "No topics found.", className="text-light")
            ])
        ])
    ])

if __name__ == "__main__":
    app.run(debug=True, port=8050)

