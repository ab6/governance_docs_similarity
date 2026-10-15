from governance_docs_similarity.datasheet_generator import generate_datasheet

datasheet = generate_datasheet(
    dataset_name="Emerging Technology Observatory AGORA dataset",
    version="1.33.0",
    description="Updated regularly, AGORA includes summaries, document text, thematic tags, and filters to help you quickly discover "
        "and analyze key developments in AI governance." 
        "The AGORA dataset consists of the csv tables documents, segments, collections, and authorities " 
        "and a folder of text documents called fulltext.",
    source="It is a collection of AI-relevant laws, regulations, standards, and other governance documents " 
        "from the United States and around the world.",
    collection_method="Document text and some metadata are taken from official sources. " 
        "Other metadata, summaries, and tags are produced by ETO analysts and annotators.",
    size="1251 text files; 4 csv tables with 1251, 1251, 1251, and 1251 rows respectively.",
    features=[
        {"name": "documents", "type": "csv", "description": "Contains metadata and summaries of governance documents."},
        {"name": "segments", "type": "csv", "description": "Contains segments of text extracted from the documents."},
        {"name": "collections", "type": "csv", "description": "Contains information about collections of documents."},
        {"name": "authorities", "type": "csv", "description": "Contains information about authorities responsible for the documents."},
        {"name": "fulltext", "type": "folder of text files", 
         "description": "Contains the full text of the governance documents in individual text files."},
    ],
    demographic_composition="No sensitive information such as personally identifiable information (PII) or protected attributes (e.g., race, gender, age) is included in the dataset.",
    known_biases="The current dataset skews toward U.S. law and policy.",
    preprocessing_steps=[
        {"annotation": "Summaries, tags, and metadata are produced by LLM, ETO analysts and annotators."},
    ],
    intended_use="AGORA fulltext files are unofficial copies of the underlying documents, collected by our screeners for research purposes and users' convenience.",
    prohibited_use="This dataset is subject to ETO's general terms of use. If you use it, please cite us. " 
        "Note that the dataset includes full text of AGORA documents, taken from sources such as government websites "
        "and repositories. Given its nature, we believe all of this material is open to non-commercial use consistent with our general terms of use, but we make no warranties.",
    retention_policy="Dataset retained indefinitely. AGORA is updated regularly, and new versions of the dataset are released periodically.",
    contact="cset@georgetown.edu",
)
    
with open("./data/datasheets/DATASHEET-AGORA.md", "w") as f:
    f.write(datasheet)