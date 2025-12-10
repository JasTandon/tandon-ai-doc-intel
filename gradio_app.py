import gradio as gr
import os
import sys
import pandas as pd
import tempfile
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from tandon_ai_doc_intel import DocumentPipeline

# --- Helper Functions ---

def process_file(file_obj, api_key):
    if not api_key:
        return None, "Error: OpenAI API Key is required."
    
    pipeline = DocumentPipeline(openai_api_key=api_key)
    
    # Gradio passes file_obj as a NamedString or wrapper, we need path
    try:
        # If multiple files, file_obj is a list
        if isinstance(file_obj, list):
            results = []
            for f in file_obj:
                res = pipeline.process(f.name)
                res.filename = os.path.basename(f.name)
                results.append(res)
            return results, "Batch processing complete."
        else:
            # Single file
            res = pipeline.process(file_obj.name)
            res.filename = os.path.basename(file_obj.name)
            return [res], "Processing complete."
    except Exception as e:
        return None, f"Error: {str(e)}"

def analyze_results(results):
    if not results:
        return pd.DataFrame(), None, None, None

    # Create Dataframe
    data = []
    embeddings_list = []
    filenames = []
    
    for r in results:
        data.append({
            "Filename": getattr(r, "filename", "Unknown"),
            "Type": "Digital" if r.metadata.get("is_digital_pdf") else "Scanned",
            "Latency (s)": r.processing_time_seconds.get("Total", 0),
            "Quality Score": r.validation_score * 100,
            "Readability": r.readability_score,
            "Gunning Fog": r.gunning_fog,
            "Sentiment": r.sentiment_polarity,
            "Risk": r.risk_analysis.get("risk_level", "Unknown") if r.risk_analysis else "Unknown",
            "Entities": len(r.entities)
        })
        if r.embeddings:
            embeddings_list.append(r.embeddings)
            filenames.append(getattr(r, "filename", "Unknown"))
            
    df = pd.DataFrame(data)
    
    # 1. Scatter Plot (Readability vs Latency)
    fig_scatter = px.scatter(df, x="Readability", y="Latency (s)", color="Type", hover_data=["Filename"], title="Latency vs. Complexity")
    
    # 2. Risk Pie Chart
    fig_pie = px.pie(df, names="Risk", title="Risk Distribution")
    
    # 3. PCA Cluster Plot
    fig_cluster = None
    if len(embeddings_list) >= 3:
        X = np.array(embeddings_list)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=min(3, len(embeddings_list)), random_state=42)
        clusters = kmeans.fit_predict(X)
        
        df_pca = pd.DataFrame({
            "x": components[:, 0],
            "y": components[:, 1],
            "Cluster": [f"Group {c+1}" for c in clusters],
            "Filename": filenames
        })
        fig_cluster = px.scatter(df_pca, x="x", y="y", color="Cluster", hover_data=["Filename"], title="Semantic Clustering (PCA)")
    
    return df, fig_scatter, fig_pie, fig_cluster

# --- UI Layout ---

def app_interface(files, api_key):
    results, msg = process_file(files, api_key)
    
    if not results:
        return msg, None, None, None, None, ""
    
    df, scatter, pie, cluster = analyze_results(results)
    
    # Generate a simple text summary
    summary_txt = f"Processed {len(results)} documents.\n"
    summary_txt += f"Average Latency: {df['Latency (s)'].mean():.2f}s\n"
    summary_txt += f"Average Readability: {df['Readability'].mean():.1f}\n"
    
    # Detailed single doc view (just taking first one as example)
    first_doc = results[0]
    detail_txt = f"--- First Document: {first_doc.filename} ---\n"
    detail_txt += f"Summary: {first_doc.summary}\n"
    if first_doc.topics:
        detail_txt += f"Topics: {', '.join(first_doc.topics)}\n"
        
    return summary_txt, df, scatter, pie, cluster, detail_txt

with gr.Blocks(title="Tandon AI Doc Intel") as demo:
    gr.Markdown("# 📄 Tandon AI Document Intelligence")
    gr.Markdown("Upload PDF documents for research-grade analysis: Latency, Readability, Risk, and Semantic Clustering.")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")
            file_input = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
            btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Status / Summary", lines=4)
    
    with gr.Tabs():
        with gr.TabItem("📊 Aggregate Data"):
            data_output = gr.DataFrame(label="Results Table")
            with gr.Row():
                scatter_plot = gr.Plot(label="Latency vs Readability")
                pie_plot = gr.Plot(label="Risk Distribution")
        
        with gr.TabItem("🧠 Semantic Intelligence"):
            cluster_plot = gr.Plot(label="Semantic Clusters (PCA)")
            
        with gr.TabItem("🔍 Inspector"):
            detail_output = gr.Textbox(label="Detailed Analysis (First Doc)", lines=10)

    btn.click(
        fn=app_interface,
        inputs=[file_input, api_input],
        outputs=[status_output, data_output, scatter_plot, pie_plot, cluster_plot, detail_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

