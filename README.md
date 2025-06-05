# EcoTalkBot

Created by Veridicity: https://veridicity.ai

EcoTalkBot is a chatbot that was developed as part of a research project at Aarhus university, which aims to promote interest in and understanding of biodiversity on agricultural land. Through an interactive dialogue with the chatbot, users have easy access to the latest knowledge about biodiversity. The answers are based on a selection of reliable sources, made avaialble through a RAG system. 
The app runs on an ubuntu machine on Azure, in a docker. It is written in Python, uses GPT-4o through the Azure API, a Weaviate vector database (locally on the cloud instance).
The main file is ecotalkbot.py
You can start the app by running 'sudo docker compose up --build -d' in the ecotalkbot directory.

## The user interface
We use Streamlit for the UI. There is a side bar on the left that provides information about the project. In the main panel the user can chat with the bot. There is a second tab where the user can view the list of source documents that the system can search in.
In the chat window a welcome message is displayed, and some queries are suggested to get started with.

## Source documents
The researchers involved in the project compiled a spreadsheet with relevant documents.
The documents were downloaded in HTML or PDF format. 


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
We use BAAI/bge-m3 FlagEmbeddings as the embedding model to encode the chunks. This enable search on semantic similarity, i.e. finding document chunks that are simlar in meaning to the user query.

### Challenges and future work
An to do item for future work is to automatically update the sources at regular time intervals if needed. 
Over the course of the project, some pages had been removed or renamed, leading to broken links in the references. This is a difficult challege to address, since ideally they should be replaces with an equivalent source, whcihc is difficult to do reliably without human intervention. However missing pages could be flagged during scheduled updated, and the researchers could be prompted to act on this. 
