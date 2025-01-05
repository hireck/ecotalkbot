import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure
import json

client = weaviate.connect_to_local()
print(client.is_ready())

client.collections.delete("DocumentChunk")

client.collections.create(
    "DocumentChunk",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="page_content", data_type=DataType.TEXT),
        Property(name="section_headers", data_type=DataType.TEXT_ARRAY),
        Property(name="parent_doc", data_type=DataType.TEXT),
        Property(name="chunk_number", data_type=DataType.INT),
        Property(name="link", data_type=DataType.TEXT),
        Property(name="target_audience", data_type=DataType.TEXT_ARRAY),
        Property(name="year", data_type=DataType.INT),
        Property(name="abstract", data_type=DataType.TEXT),
        Property(name="keywords", data_type=DataType.TEXT_ARRAY),
        Property(name="data_type", data_type=DataType.TEXT),
        Property(name="type_of_information", data_type=DataType.TEXT),
        Property(name="geography", data_type=DataType.TEXT_ARRAY),
        Property(name="language", data_type=DataType.TEXT),
        Property(name="publisher", data_type=DataType.TEXT),
        Property(name="author", data_type=DataType.TEXT),
        Property(name="open_access", data_type=DataType.BOOL),
        Property(name="available_as_pdf", data_type=DataType.BOOL),
    ],
    vectorizer_config=Configure.Vectorizer.none(),
)
client.close()