import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from tandon_ai_doc_intel import DocumentPipeline

st.set_page_config(
    page_title="Tandon AI Doc Intel",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 Tandon AI Document Intelligence")
    st.markdown("""
    **Research-Grade Unstructured Document Analytics**  
    Upload documents (PDF) to classify, extract, enrich, validate, and benchmark.
    """)

    # --- Sidebar: Configuration ---
    st.sidebar.header("Configuration")
    
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.sidebar.warning("⚠️ Please provide an OpenAI API Key to use Enrichment features.")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Pipeline Stages:**\n"
        "1. Ingestion\n"
        "2. Classification (Digital/Scanned)\n"
        "3. Extraction (Hybrid)\n"
        "4. Enrichment (LLM)\n"
        "5. Validation (Quality)\n"
        "6. Analytics (ML & Metrics)\n"
        "7. Vector Storage"
    )

    # --- Main: File Upload ---
    uploaded_files = st.file_uploader("Upload PDF Document(s)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Analyze Documents"):
        if not api_key:
            st.error("OpenAI API Key is required for the full pipeline.")
            return
            
        if len(uploaded_files) > 100:
            st.error("Maximum 100 files allowed.")
            return

        with st.spinner("Initializing Pipeline..."):
            pipeline = DocumentPipeline(openai_api_key=api_key)

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_size = 10
        total_files = len(uploaded_files)
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = uploaded_files[i : i + batch_size]
            status_text.text(f"Processing batch {i//batch_size + 1} ({len(batch)} files)...")
            
            for file_idx, uploaded_file in enumerate(batch):
                # Update progress
                current_progress = (i + file_idx) / total_files
                progress_bar.progress(current_progress)
                
                # Save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    result = pipeline.process(tmp_path)
                    result.filename = uploaded_file.name # Add filename to result for display
                    results.append(result)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
        progress_bar.progress(100)
        status_text.empty()
        st.success("Processing Complete!")

        # --- Dashboard ---
        if len(results) == 1:
            render_single_document_view(results[0])
        else:
            render_aggregate_view(results)

def render_single_document_view(result):
    # 1. Scientific Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Document Type", "Digital PDF" if result.metadata.get("is_digital_pdf") else "Scanned PDF")
    
    with col2:
        # Flesch Reading Ease
        score = result.readability_score
        label = "Easy" if score > 60 else "Complex" if score > 30 else "Very Complex"
        st.metric("Readability (Flesch)", f"{score:.1f}", label)
    
    with col3:
        # Validation
        v_score = result.validation_score * 100
        st.metric("Data Quality Score", f"{v_score:.1f}%")

    with col4:
        risk = "Unknown"
        if result.risk_analysis:
            risk = result.risk_analysis.get("risk_level", "Unknown")
        st.metric("Risk Assessment", risk)
        
    with col5:
        total_time = result.processing_time_seconds.get("Total", 0)
        st.metric("Total Latency", f"{total_time:.2f}s")

    st.divider()

    # 2. Detailed Analytics Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Performance Metrics", 
        "🧠 Machine Learning", 
        "📝 Content Analysis", 
        "📊 Structured Data", 
        "✅ Validation", 
        "📄 Raw Text"
    ])

    with tab1:
        st.subheader("Pipeline Latency Breakdown")
        if result.processing_time_seconds:
            timings = {k: v for k, v in result.processing_time_seconds.items() if k != "Total"}
            df_time = pd.DataFrame(list(timings.items()), columns=["Stage", "Time (s)"])
            
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df_time, x="Stage", y="Time (s)", title="Processing Time per Stage", color="Stage")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                    st.dataframe(df_time, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Topic Modeling (NMF)")
            if result.topics:
                for t in result.topics:
                    st.markdown(f"- **{t}**")
            else:
                st.info("Not enough text chunks.")

            st.subheader("Keyphrase Extraction (TF-IDF)")
            if result.ml_keywords:
                st.write(", ".join([f"`{k}`" for k in result.ml_keywords]))
        
        with c2:
            st.subheader("Semantic Metrics")
            st.metric("Sentiment Polarity", f"{result.sentiment_polarity:.2f}", help="-1 (Neg) to +1 (Pos)")
            st.metric("Subjectivity", f"{result.sentiment_subjectivity:.2f}", help="0 (Obj) to 1 (Subj)")
            st.metric("Lexical Diversity (TTR)", f"{result.lexical_diversity:.2f}", help="Unique Words / Total Words")

            st.subheader("Entity Extraction (LLM)")
            if result.entities:
                df_entities = pd.DataFrame(result.entities)
                if "type" in df_entities.columns:
                        fig = px.pie(df_entities, names="type", title="Entity Types")
                        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Executive Summary")
        if result.summary:
            st.info(result.summary)
        
        st.subheader("Risk Analysis")
        if result.risk_analysis:
            risk_data = result.risk_analysis
            st.write(f"**Level:** {risk_data.get('risk_level')}")
            factors = risk_data.get("risk_factors", [])
            if factors:
                for f in factors:
                    st.markdown(f"- ⚠️ {f}")

    with tab4:
        st.subheader("Extracted Tables")
        if result.tables:
            st.write(f"Found {len(result.tables)} tables.")
            for i, table in enumerate(result.tables):
                with st.expander(f"Table {i+1} (Page {table.get('page')}) - Accuracy: {table.get('accuracy', 0):.1f}%"):
                    if "data" in table:
                        st.dataframe(pd.DataFrame(table["data"]))
        else:
            st.info("No tables detected.")

    with tab5:
        st.subheader("Quality Validation")
        if result.validation_issues:
            st.error("Issues Detected:")
            for issue in result.validation_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No validation issues detected. Document quality is high.")

    with tab6:
        st.text_area("Full Text", result.text, height=400)


def render_aggregate_view(results):
    st.header(f"Aggregate Analysis ({len(results)} Documents)")
    
    # Create main dataframe
    data = []
    for r in results:
        data.append({
            "Filename": getattr(r, "filename", "Unknown"),
            "Type": "Digital" if r.metadata.get("is_digital_pdf") else "Scanned",
            "Latency (s)": r.processing_time_seconds.get("Total", 0),
            "Quality Score": r.validation_score * 100,
            "Readability": r.readability_score,
            "Sentiment": r.sentiment_polarity,
            "Risk": r.risk_analysis.get("risk_level", "Unknown") if r.risk_analysis else "Unknown",
            "Tables": len(r.tables),
            "Entities": len(r.entities)
        })
    df = pd.DataFrame(data)
    
    # Top Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Latency", f"{df['Latency (s)'].mean():.2f}s")
    with c2:
        st.metric("Avg Quality", f"{df['Quality Score'].mean():.1f}%")
    with c3:
        st.metric("Avg Readability", f"{df['Readability'].mean():.1f}")
    with c4:
        high_risk_count = df[df['Risk'] == 'High'].shape[0]
        st.metric("High Risk Docs", high_risk_count)
        
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latency vs Readability")
        fig = px.scatter(df, x="Readability", y="Latency (s)", color="Type", hover_data=["Filename"])
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Risk Distribution")
        fig = px.pie(df, names="Risk", title="Risk Levels")
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Detailed Results Table")
    st.dataframe(df, use_container_width=True)
    
    # Export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", csv, "batch_results.csv", "text/csv")

if __name__ == "__main__":
    main()
