import chromadb

client = chromadb.Client()
collection = client.create_collection(name="test")

collection.add(
    documents=[
        "Applicants must provide income proof for loans above AED 50,000",
        "KYC verification requires Emirates ID and a valid passport",
        "Loan disbursal happens within 48 hours of final approval"
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
    query_texts=["what salary documents do I need"],
    n_results=1
)

print(results["documents"])
print(results["distances"])
