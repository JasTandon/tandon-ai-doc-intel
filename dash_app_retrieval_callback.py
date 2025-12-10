
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
    
    # We need to rebuild a temporary vector store from the stored data
    # because we don't have a persistent server-side vector store in this simple app demo.
    # In a real app, this would query ChromaDB directly.
    
    from tandon_ai_doc_intel.embeddings import OpenAIEmbeddings
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        # 1. Embed Query
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(input=query, model="text-embedding-ada-002")
        query_vec = resp.data[0].embedding
        
        # 2. Brute-force search over stored document embeddings (doc-level) 
        # Note: Ideally we search chunks, but we only stored doc-level average embeddings in 'stored-data' to save space.
        # For a better demo, we should have stored chunks. 
        # But 'stored-data' has 'embeddings' field which is doc-level.
        
        results = []
        for d in data:
            if d.get('embeddings'):
                score = cosine_similarity([query_vec], [d['embeddings']])[0][0]
                results.append((score, d))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        top_k = results[:5]
        
        if not top_k:
            return html.Div("No embeddings found in processed documents.", className="text-muted")
            
        cards = []
        for score, doc in top_k:
            cards.append(
                dbc.Card([
                    dbc.CardHeader(html.Div([
                        html.Strong(doc['filename']),
                        html.Badge(f"{score:.4f}", color="info", className="ms-2")
                    ], className="d-flex justify-content-between align-items-center"), className="text-light"),
                    dbc.CardBody([
                        html.P(doc['summary'][:200] + "..." if doc['summary'] else "No summary available.", className="text-light small")
                    ])
                ], className="mb-2 bg-dark border-secondary")
            )
            
        return html.Div(cards)
        
    except Exception as e:
        return html.Div(f"Retrieval Error: {str(e)}", className="text-danger")

