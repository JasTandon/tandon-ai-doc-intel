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
    Upload a document (PDF) to classify, extract, enrich, validate, and benchmark.
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
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file and st.button("Analyze Document"):
        if not api_key:
            st.error("OpenAI API Key is required for the full pipeline.")
            return

        with st.spinner("Initializing Pipeline..."):
            pipeline = DocumentPipeline(openai_api_key=api_key)

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # --- Processing ---
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.text("Processing Document Pipeline...")
            progress_bar.progress(30)
            
            result = pipeline.process(tmp_path)
            
            progress_bar.progress(100)
            status_text.empty()
            
            st.success("Processing Complete!")

            # --- Dashboard ---
            
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
                    # Filter out 'Total' for the pie chart
                    timings = {k: v for k, v in result.processing_time_seconds.items() if k != "Total"}
                    df_time = pd.DataFrame(list(timings.items()), columns=["Stage", "Time (s)"])
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(df_time, x="Stage", y="Time (s)", title="Processing Time per Stage", color="Stage")
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                         st.dataframe(df_time, use_container_width=True)
                else:
                    st.info("No timing data available.")

            with tab2:
                st.subheader("Topic Modeling (NMF)")
                if result.topics:
                    for t in result.topics:
                        st.markdown(f"- **{t}**")
                else:
                    st.info("Not enough text chunks to perform topic modeling.")

                st.subheader("Keyphrase Extraction (TF-IDF)")
                if result.ml_keywords:
                    st.write(", ".join([f"`{k}`" for k in result.ml_keywords]))
                else:
                    st.info("No keywords extracted.")
                    
                st.subheader("Entity Extraction (LLM)")
                if result.entities:
                    df_entities = pd.DataFrame(result.entities)
                    if "type" in df_entities.columns:
                         fig = px.pie(df_entities, names="type", title="Entity Types Distribution")
                         st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No entities found.")

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

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    main()
