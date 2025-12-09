import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from tandon_ai_doc_intel import DocumentPipeline

def main():
    # 1. Initialize the pipeline
    # You need to set OPENAI_API_KEY env var or pass it here
    api_key = os.getenv("OPENAI_API_KEY", "your-key-here")
    pipeline = DocumentPipeline(openai_api_key=api_key)

    # 2. Define a document source (path to a PDF)
    # Replace with a real PDF path
    doc_path = "sample_invoice.pdf"

    # Check if file exists for demo purposes
    if not os.path.exists(doc_path):
        print(f"File {doc_path} not found. Please provide a valid PDF.")
        # Create a dummy file for demonstration if it doesn't exist? 
        # No, better to just warn.
        return

    print(f"Processing {doc_path}...")

    # 3. Process the document
    try:
        result = pipeline.process(doc_path)

        # 4. Inspect results
        print("\n--- Processing Complete ---")
        print(f"Digital PDF: {result.metadata.get('is_digital_pdf')}")
        print(f"Text Length: {len(result.text)} chars")
        print(f"Summary: {result.summary}")
        print(f"Entities Found: {len(result.entities)}")
        
        if result.tables:
            print(f"Tables Found: {len(result.tables)}")
            # Print first table's simple structure or shape if available
            print(f"Table 1 info: {result.tables[0].get('accuracy')}% accuracy")

        # Validation Logic
        print(f"Validation Score: {result.validation_score:.2f}")
        if result.validation_issues:
            print("Validation Issues Found:")
            for issue in result.validation_issues:
                print(f" - {issue}")
                
        if result.risk_analysis:
            print(f"Risk Level: {result.risk_analysis.get('risk_level')}")
        
        print("\n--- Sample Text ---")
        print(result.text[:200] + "...")

    except Exception as e:
        print(f"Error processing document: {e}")

if __name__ == "__main__":
    main()
