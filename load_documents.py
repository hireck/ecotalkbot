import weaviate
import json

client = weaviate.connect_to_local()
print(client.is_ready())
chunks = client.collections.get("DocumentChunk")

with open("doc_chunks.json") as f:
    data = json.load(f)
    for d in data:
        #print(d["page_content"])
        #print(d["bge_dense_vector"])

        # Build the object payload
        chunk_obj = {
            "title": d["title"],
            "page_content": d["page_content"],
            "section_headers": d.get("section_headers"),
            "parent_doc": d["parent_doc"],
            "chunk_number": int(d["chunk_number"]),
            "link": d.get("link"),
            "target_audience": d.get("target_audience"),
            "year": d.get("year"),
            "abstract": d.get("abstract"),
            "keywords": d.get("keywords"),
            "data_type": d.get("data_type"),
            "type_of_information": d.get("type_of_information"),
            "geography": d.get("geography"),
            "language": d.get("language"),
            "publisher": d.get("publisher"),
            "author": d.get("author"),
            "open_access": d.get("open_access"),
            "available_as_pdf": d.get("available_as_pdf"),
        }
        print(chunk_obj)
        # Get the vector
        vector = d["bge_dense_vector"]

        # Add object (including vector) 
        uuid = chunks.data.insert(
            properties=chunk_obj,
            vector=vector  # Add the custom vector
            # references=reference_obj  # You can add references here
        )
client.close()