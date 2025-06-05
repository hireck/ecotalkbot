# EcoTalkBot

Created by Veridicity: https://veridicity.ai

EcoTalkBot is a chatbot that was developed as part of a research project at Aarhus university, which aims to promote interest in and understanding of biodiversity on agricultural land. Through an interactive dialogue with the chatbot, users have easy access to the latest knowledge about biodiversity. The answers are based on a selection of reliable sources, made avaialble through a RAG system. 
The app runs on an ubuntu machine on Azure, in a docker. It is written in Python, uses GPT-4o through the Azure API, a Weaviate vector database (locally on the cloud instance).
The main file is ecotalkbot.py
You can start the app by running 'sudo docker compose up --build -d' in the ecotalkbot directory.

## The user interface
We use Streamlit for the UI. There is a side bar on the left that provides information about the project. In the main panel the user can chat with the bot. There is a second tab where the user can view the list of source documents that the system can search in.
In the chat window a welcome message is displayed, and some queries are suggested to get started with.

The information displayed is in Danish. In future interations we plan to enable switching to other languages

## Source documents
The researchers involved in the project compiled a spreadsheet with relevant documents.
The documents were downloaded in HTML or PDF format. Many of the documents are written in Danish, but some documents in English, German, and Dutch were also included.


### Processing html documents
* Some webpages were in javascript and needed to be rendered in order to get the html. We used the selenium webdriver package to do this.
* For some documents on the list, also subpages were downloaded, following links with a specific prefix. 
* We then used the HTMLHeaderTextSplitter from LangChain to split each document into chunks. This uses the html header tags to identify sections, producing nicely coherent chunks, with the header as metadata. (Unfortunately, the HTMLHeaderTextSplitter also extracts each header as its own section, so we need to filter those out.)
* We add the information from the original spreadsheet to the chunks as metadata

### Processing PDF documents
* PDF documents were converted to markdown with marker pdf
* We then use the MarkdownHeaderTextSplitter from LangChain to split the content into sections with headers.
* We add the information from the original spreadsheet to the chunks as metadata

### Uploading the chunks to a Weaviate cluster
We use BAAI/bge-m3 FlagEmbeddings as the embedding model to encode the chunks. This enables search on semantic similarity, i.e. finding document chunks that are simlar in meaning to the user query.

### Challenges and future work
An to do item for future work is to automatically update the sources at regular time intervals if needed. 
Over the course of the project, some pages had been removed or renamed, leading to broken links in the references. This is a difficult challege to address, since ideally they should be replaces with an equivalent source, whcihc is difficult to do reliably without human intervention. However missing pages could be flagged during scheduled updated, and the researchers could be prompted to act on this. 

## Retrieval
The retrieval component of a RAG system is like a search engine that finds the most relevant document chunks to provide the Large Language Model (LLM) (in this case GPT-4o) with the information it needs to respond to the user query.
For this initial implementation, we only used vector search, using the BAAI/bge-m3 FlagEmbeddings model. We chose this model, because it is suitable for multi-lingual retrieval. Other multi-lingual embedding models to try in the future inlcude EuroBERT and multilingual-e5-large-instruct.
The 28GB Azure instance we had for this project turned out to be on the small side for running an embedding model of this size. Therefore we kept the retrieval limited to one step and refrained from using a reranker. A reranker would, among other things, facilitate combining vector search with keyword search.

### Contextualizing the query
Rather than using the original user query, we prompt the LLM to rephrase the query, using the preceding conversation as context. This is because soemtimes queries lack information, for example they will use words like 'it' to refer to something that was mentionaed earlier. The contextualizing step serves to include such information in the query explicitly so it can be understood without context.

> Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. The overall topic is biodiversity.
> Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Keep it in the original language.
>
> Chat history:
> {history}
>
> Latest user question:
> {question}
>
> Standalone version of the question:

## Response generation

## Data collection
