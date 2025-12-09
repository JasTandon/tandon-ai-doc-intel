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
    **Production-Ready Unstructured Document Analytics**  
    Upload a document (PDF) to classify, extract, enrich, and validate using the pipeline.
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
        "**Pipeline Features:**\n"
        "- Auto-Classification (Digital/Scanned)\n"
        "- Hybrid Extraction (PyMuPDF / OCR)\n"
        "- Table Extraction (Camelot)\n"
        "- LLM Enrichment (Summaries, Risks)\n"
        "- Quality Validation Loop"
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

            start_time = time.time()
            
            status_text.text("Ingesting & Classifying...")
            progress_bar.progress(20)
            
            # We can't easily hook into the pipeline steps without callbacks, 
            # so we'll just run the whole process and measure total time.
            # In a real async UI, we'd use callbacks.
            
            status_text.text("Extracting, Enriching, & Validating... (This may take a moment)")
            progress_bar.progress(50)
            
            result = pipeline.process(tmp_path)
            
            end_time = time.time()
            duration = end_time - start_time
            
            progress_bar.progress(100)
            status_text.empty() # Clear status
            
            st.success(f"Processing Complete in {duration:.2f} seconds!")

            # --- Dashboard ---
            
            # 1. Top Level Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Document Type", "Digital PDF" if result.metadata.get("is_digital_pdf") else "Scanned PDF")
            
            with col2:
                score = result.validation_score * 100
                st.metric("Validation Score", f"{score:.1f}%", delta_color="normal" if score > 80 else "inverse")
            
            with col3:
                risk = "Unknown"
                if result.risk_analysis:
                    risk = result.risk_analysis.get("risk_level", "Unknown")
                
                risk_color = "normal"
                if risk.lower() == "high": risk_color = "inverse"
                st.metric("Risk Level", risk)

            with col4:
                st.metric("Entities Found", len(result.entities))

            st.divider()

            # 2. Detailed Analytics Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Summary & Risk", "🔍 Entities", "📊 Tables", "✅ Validation", "📄 Raw Content"])

            with tab1:
                st.subheader("Executive Summary")
                if result.summary:
                    st.info(result.summary)
                else:
                    st.text("No summary generated.")

                st.subheader("Risk Analysis")
                if result.risk_analysis:
                    risk_data = result.risk_analysis
                    st.write(f"**Level:** {risk_data.get('risk_level')}")
                    
                    factors = risk_data.get("risk_factors", [])
                    if factors:
                        st.write("**Risk Factors:**")
                        for f in factors:
                            st.markdown(f"- ⚠️ {f}")
                else:
                    st.text("No risk analysis available.")

            with tab2:
                st.subheader("Extracted Entities")
                if result.entities:
                    # Create DataFrame for better display
                    df_entities = pd.DataFrame(result.entities)
                    
                    # Chart: Entity Types Distribution
                    if "type" in df_entities.columns:
                        type_counts = df_entities["type"].value_counts().reset_index()
                        type_counts.columns = ["Entity Type", "Count"]
                        fig = px.bar(type_counts, x="Entity Type", y="Count", title="Entity Distribution", color="Entity Type")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df_entities, use_container_width=True)
                else:
                    st.text("No entities found.")

            with tab3:
                st.subheader("Extracted Tables")
                if result.tables:
                    st.write(f"Found {len(result.tables)} tables.")
                    for i, table in enumerate(result.tables):
                        with st.expander(f"Table {i+1} (Page {table.get('page')}) - Accuracy: {table.get('accuracy', 0):.1f}%"):
                            # 'data' is a list of dicts, convert to DF
                            if "data" in table:
                                st.dataframe(pd.DataFrame(table["data"]))
                else:
                    st.info("No tables detected in this document.")

            with tab4:
                st.subheader("Quality Validation")
                score_pct = result.validation_score * 100
                st.progress(result.validation_score)
                st.write(f"**Confidence Score:** {score_pct:.1f}/100")
                
                if result.validation_issues:
                    st.error("Issues Detected:")
                    for issue in result.validation_issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("No validation issues detected. Document quality is high.")

            with tab5:
                st.subheader("Raw Text Content")
                st.text_area("Full Text", result.text, height=400)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    main()

