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
from tandon_ai_doc_intel.embeddings import VectorStore, OpenAIEmbeddings

# --- Cache Setup for Background Callbacks ---
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP], 
                suppress_callback_exceptions=True, 
                background_callback_manager=background_callback_manager)
app.title = "Tandon AI Doc Intel"

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('brown')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

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
        "cluster": f"Interpret these document clusters found via PCA/K-Means: {context_data}. Suggest potential thematic groupings.",
        "technical": f"Interpret these technical text metrics: {context_data}. Explain what they imply about the document's structure and density."
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
        # Extra robust cleanup to prevent stuck file handles
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

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
                dbc.Tab(label="🔎 Retrieval Lab", tab_id="tab-4", label_style={"color": "#adb5bd"}, active_label_style={"color": "#fff", "fontWeight": "bold"}),
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
    
    # Pre-process: Decode all files to check properties
    import fitz
    
    single_page_docs = []
    multi_page_docs = []
    
    logger.info("Starting ingestion phase...")
    
    for c, n in zip(list_of_contents, list_of_names):
        try:
            content_type, content_string = c.split(',')
            decoded = base64.b64decode(content_string)
            
            with fitz.open(stream=decoded, filetype="pdf") as doc:
                page_count = len(doc)
                
            item = {"filename": n, "content": c, "pages": page_count}
            if page_count == 1:
                single_page_docs.append(item)
            else:
                multi_page_docs.append(item)
        except Exception as e:
            logger.error(f"Failed to check page count for {n}: {e}")
            # Treat as multi-page (safe fallback)
            multi_page_docs.append({"filename": n, "content": c, "pages": 1})

    total_docs = len(single_page_docs) + len(multi_page_docs)
    processed_count = 0
    
    # 1. Process Multi-Page Docs (One by One)
    for doc in multi_page_docs:
        percent = int((processed_count / total_docs) * 100) if total_docs > 0 else 0
        set_progress((str(percent), f"{percent}%"))
        
        n = doc['filename']
        c = doc['content']
        logger.info(f"Processing Multi-Page Document: {n} ({doc['pages']} pages)")
        
        res, err = parse_contents(c, n, api_key)
        processed_count += 1
        
        # Always update progress after processing
        percent = int((processed_count / total_docs) * 100) if total_docs > 0 else 0
        set_progress((str(percent), f"{percent}%"))
        
        if err:
            logger.error(f"Skipping {n} due to error: {err}")
            continue
            
        results_data.append(_serialize_result(res))

    # 2. Process Single-Page Docs (Batched in groups of 3)
    batch_size = 3
    for i in range(0, len(single_page_docs), batch_size):
        batch = single_page_docs[i : i + batch_size]
        
        percent = int((processed_count / total_docs) * 100) if total_docs > 0 else 0
        set_progress((str(percent), f"{percent}%"))
        
        logger.info(f"Processing Batch of {len(batch)} Single-Page Docs: {[d['filename'] for d in batch]}")
        
        # Use ThreadPoolExecutor to process batch in parallel
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_doc = {
                executor.submit(parse_contents, d['content'], d['filename'], api_key): d 
                for d in batch
            }
            
            for future in concurrent.futures.as_completed(future_to_doc):
                d = future_to_doc[future]
                processed_count += 1
                
                # Update progress for each item in batch
                percent = int((processed_count / total_docs) * 100) if total_docs > 0 else 0
                set_progress((str(percent), f"{percent}%"))
                
                try:
                    res, err = future.result()
                    if err:
                        logger.error(f"Error in batch for {d['filename']}: {err}")
                    else:
                        results_data.append(_serialize_result(res))
                except Exception as e:
                    logger.error(f"Crash in batch for {d['filename']}: {e}")

    if not results_data and total_docs > 0:
         return [], "⚠️ All files failed. Check API Key or console logs."

    # Final progress update to 100% ONLY after loop is done
    set_progress(("100", "100%"))
    return results_data, "" # Clear loading text when done

def _serialize_result(res):
    return {
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
        "topics": res.topics,
        "info_density": res.info_density,
        "entity_density": res.entity_density,
        "sentence_complexity": res.sentence_complexity,
        "cost_estimate": res.cost_estimate_usd,
        "token_usage": res.token_usage,
        "factuality_score": res.factuality_score,
        "ocr_cer": res.ocr_cer,
        "ocr_wer": res.ocr_wer
    }

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
    total_processing_time = df['timings'].apply(lambda x: x.get('Total', 0)).sum()
    avg_readability = df['readability'].mean()
    high_risk = len(df[df['risk'] == 'High'])
    throughput = len(df) / total_processing_time if total_processing_time > 0 else 0.0
    
    return html.Div([
        html.H6(f"Docs Processed: {len(df)}"),
        html.Hr(className="my-2"),
        html.P(f"⏱ Avg Latency: {avg_latency:.2f}s"),
        html.P(f"⚡ Throughput: {throughput:.2f} docs/s"),
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
                                 title="Complexity vs. Performance (Digital=Native PDF, Scanned=OCR)", template="plotly_dark")
        
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
                dbc.Col(dbc.Card([
                    dbc.CardHeader("🤖 AI Risk Assessment", style={"color": "white"}), 
                    dbc.CardBody(ai_risk_msg, style={"color": "white"})
                ], className="bg-dark border-info mb-3"), width=12)
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
                dbc.Card([
                    dbc.CardHeader("🤖 AI Cluster Analysis", style={"color": "white"}), 
                    dbc.CardBody(ai_cluster_msg, style={"color": "white"})
                ], className="bg-dark border-info mb-3"),
                dcc.Graph(figure=fig_cluster)
            ])

        return html.Div([
            dbc.Row([
                dbc.Col(cluster_graph, width=8),
                dbc.Col([
                    html.H5("ML Metrics Reference", style={"color": "white"}),
                    html.Ul([
                        html.Li(f"Avg Sentiment: {df['sentiment'].mean():.2f} (-1 to 1)", style={"color": "white"}),
                        html.Li(f"Avg Subjectivity: {df['subjectivity'].mean():.2f} (0 to 1)", style={"color": "white"}),
                        html.Li(f"Avg Gunning Fog: {df['gunning_fog'].mean():.1f} (Grade Level)", style={"color": "white"}),
                        html.Li(f"Avg Lexical Diversity: {df['lexical_diversity'].mean():.2f}", style={"color": "white"})
                    ], style={"color": "white"})
                ], width=4)
            ])
        ])

    elif active_tab == "tab-4":
        # --- RETRIEVAL LAB ---
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("🔎 Semantic Search & Retrieval Lab", className="text-light mb-4"),
                    html.P("Test the vector store retrieval capabilities. Note: Formal metrics (Recall@k, MRR) require ground truth queries and should be run via the CLI benchmarking script.", className="text-muted mb-4"),
                    
                    dbc.InputGroup([
                        dbc.Input(id="search-query", placeholder="Enter semantic query (e.g., 'financial risks in Q3')...", type="text"),
                        dbc.Button("Search", id="btn-search", color="primary")
                    ], className="mb-4"),
                    
                    dbc.Spinner(html.Div(id="search-results"), color="light", size="sm")
                ], width=8, className="mx-auto")
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
    State("stored-data", "data"),
    State("api-key", "value") # Include API key in state
)
def update_inspector(selected_idx, data, api_key):
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
    
    # Fix height expansion issue - Use fixed height layout
    fig_time.update_layout(
        autosize=False,
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
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
                
    # New Technical Metrics Card
    dbc.Card([
        dbc.CardHeader("🔬 Advanced ML Metrics", className="text-light"),
        dbc.ListGroup([
            dbc.ListGroupItem(f"Info Density: {doc.get('info_density', 0.0):.2f}", className="bg-dark text-light"),
            dbc.ListGroupItem(f"Entity Density: {doc.get('entity_density', 0.0):.2f}", className="bg-dark text-light"),
            dbc.ListGroupItem(f"Sentence Complexity (StdDev): {doc.get('sentence_complexity', 0.0):.2f}", className="bg-dark text-light"),
        ], flush=True),
        dbc.CardBody(
            get_ai_insight({
                "info_density": doc.get('info_density', 0), 
                "entity_density": doc.get('entity_density', 0),
                "complexity": doc.get('sentence_complexity', 0)
            }, "technical", api_key) if api_key else "⚠️ OpenAI API Key required for technical insights.",
            className="text-light small fst-italic"
        )
    ], className="mb-3 bg-dark border-info"),

    # Evaluation Metrics Card
    dbc.Card([
        dbc.CardHeader("📑 Cost Analysis and Factuality Score Metrics", className="text-light"),
        dbc.ListGroup([
            dbc.ListGroupItem(f"💰 Est. Cost: ${doc.get('cost_estimate', 0.0):.6f}", className="bg-dark text-light"),
            dbc.ListGroupItem(
                html.Div([
                    html.Span("🔢 Token Usage: "),
                    html.Span(f"In: {doc.get('token_usage', {}).get('llm_input', 0)} | Out: {doc.get('token_usage', {}).get('llm_output', 0)}", className="text-muted ms-1")
                ]), 
                className="bg-dark text-light"
            ),
            dbc.ListGroupItem(f"✅ Factuality Score: {doc.get('factuality_score', 0.0):.2f}", className="bg-dark text-light", id="tooltip-factuality"),
            
            # OCR Metrics (CER/WER) - Show N/A if not available
            dbc.ListGroupItem(
                f"📉 OCR CER: {doc.get('ocr_cer'):.4f}" if doc.get('ocr_cer') is not None else "📉 OCR CER: N/A (No Ground Truth)", 
                className="bg-dark text-light"
            ),
            dbc.ListGroupItem(
                f"📉 OCR WER: {doc.get('ocr_wer'):.4f}" if doc.get('ocr_wer') is not None else "📉 OCR WER: N/A (No Ground Truth)", 
                className="bg-dark text-light"
            ),
        ], flush=True),
        dbc.Tooltip("Proxy measure: Overlap between summary and source chunks.", target="tooltip-factuality"),
    ], className="mb-3 bg-dark border-primary"),


                dcc.Graph(figure=fig_time)
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5("🏷️ Extracted Topics", style={"color": "white"}),
                html.P(", ".join(doc.get('topics', [])) if doc.get('topics') else "No topics found.", style={"color": "white"})
            ])
        ])
    ])

if __name__ == "__main__":
    app.run(debug=True, port=8050)

# 6. Retrieval Callback
@callback(
    Output("search-results", "children"),
    Input("btn-search", "n_clicks"),
    State("search-query", "value"),
    State("stored-data", "data"),
    State("api-key", "value"),
    prevent_initial_call=True
)
def run_retrieval(n_clicks, query, data, api_key):
    if not query or not api_key:
        return html.Div("Please enter a query and ensure API Key is set.", className="text-warning")
    
    try:
        # 1. Initialize Components
        embedder = OpenAIEmbeddings(api_key=api_key)
        vector_store = VectorStore() # Defaults to ./chroma_db
        
        # 2. Embed Query
        # OpenAIEmbeddings.embed takes a list of strings
        # We need a single vector, but the provider returns List[List[float]]
        query_vecs = embedder.embed([query])
        if not query_vecs:
            return html.Div("Failed to generate embedding for query.", className="text-danger")
        query_vec = query_vecs[0]
        
        # 3. Query Vector Store
        results = vector_store.query(query_vec, n_results=5)
        
        # Results format: {'ids': [['id1', ...]], 'distances': [[0.1, ...]], 'documents': [['text', ...]], 'metadatas': [[{...}, ...]]}
        
        if not results or not results['ids'] or not results['ids'][0]:
            return html.Div("No relevant documents found in local vector store.", className="text-muted")
            
        ids = results['ids'][0]
        distances = results['distances'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
        
        cards = []
        for i in range(len(ids)):
            score = 1.0 - distances[i] # Convert distance to similarity score approx
            text_chunk = documents[i]
            meta = metadatas[i]
            
            # Try to find filename from metadata or ID
            # Our pipeline stores source_id, chunk_index. It doesn't explicitly store filename in metadata currently (oops).
            # But we can try to look it up from 'data' if we had a mapping. 
            # For now, we'll just show the chunk text.
            
            cards.append(
                dbc.Card([
                    dbc.CardHeader(html.Div([
                        html.Strong(f"Result {i+1}"),
                        html.Badge(f"Score: {score:.4f}", color="info", className="ms-2")
                    ], className="d-flex justify-content-between align-items-center"), className="text-light"),
                    dbc.CardBody([
                        html.P(text_chunk, className="text-light small")
                    ])
                ], className="mb-2 bg-dark border-secondary")
            )
            
        return html.Div(cards)
        
    except Exception as e:
        return html.Div(f"Retrieval Error: {str(e)}", className="text-danger")

