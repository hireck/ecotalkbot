# EcoTalkBot

Created by Veridicity: https://veridicity.ai

EcoTalkBot is a chatbot that was developed as part of a research project at Aarhus university, which aims to promote interest in and understanding of biodiversity on agricultural land. Through an interactive dialogue with the chatbot, users have easy access to the latest knowledge about biodiversity. The answers are based on a selection of reliable sources, made avaialble through a RAG system. 
The app runs on an ubuntu machine on Azure, in a docker. It is written in Python, uses GPT-4o through the Azure API, a Weaviate vector database (locally on the cloud instance).

The main file is ecotalkbot.py

You can start the app by running 'sudo docker compose up --build -d' in the ecotalkbot directory. After the project is built the embedding model will load once the first user logged in. This takes some time.

## The user interface
We use Streamlit for the UI. There is a side bar on the left that provides information about the project. In the main panel the user can chat with the bot. There is a second tab where the user can view the list of source documents that the system can search in.
In the chat window a welcome message is displayed, and some queries are suggested to get started with.

The information displayed is in Danish. In future interations we plan to enable switching to other languages.

![image](https://github.com/user-attachments/assets/5b20a3e5-7e16-4849-bb26-484350532172)


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
The resulting chunks with their embeddings and metadata (from the original spredsheet) are written to a json file. Here is an example of such a chunk:

```
  {
    "page_content": "Het is belangrijk dat de PARTRIDGE maatregelen niet worden bereden door landbouwmachines of worden betreden door loslopende honden. In de lente en zomer wordt hier gebroed door patrijzen en andere vogels. In de wintermaanden zijn dit de schaarse plekken in het landschap waar vogels en kleine zoogdieren veilig kunnen rusten en eten.  \n![](_page_1_Picture_11.jpeg)",
    "parent_doc": "72",
    "chunk_number": "10",
    "title": "Samenhang (Cohesive measures)",
    "section_headers": [
        "RUST"
    ],
    "link": "https://northsearegion.eu/media/22559/nl-factsheet-1-samenhang.pdf",
    "abstract": "Vejledning",
    "keywords": [
        "agerh\u00f8ne",
        "sammenh\u00e6ng"
    ],
    "data_type": "pdf",
    "type_of_information": "Vejledning, anbefaling",
    "target_audience": [
        "farmer"
    ],
    "geography": [
        "Europe",
        "Netherlands"
    ],
    "language": "Dutch",
    "publisher": "Interreg North Sea Region PARTRIDGE project (publication)",
    "author": "Interreg PARTRIDGE",
    "open_access": true,
    "available_as_pdf": true,
    "bge_dense_vector": [0.03424072265625, 0.045379638671875, -0.07666015625, 0.003021240234375, -0.0300445556640625, ... ]
  }
```
Our Weaviate cluster is part of the docker setup, running locally on our Azure instance.

The scripts setup_collection.py and load_documents.py are used to create a collection on the cluster with the properties we need and then populate it with the document chunks.


### Challenges and future work
An to do item for future work is to automatically update the sources at regular time intervals if needed. 
Over the course of the project, some pages had been removed or renamed, leading to broken links in the references. This is a difficult challege to address, since ideally they should be replaces with an equivalent source, whcihc is difficult to do reliably without human intervention. However missing pages could be flagged during scheduled updated, and the researchers could be prompted to act on this. 

## Retrieval
The retrieval component of a RAG system is like a search engine that finds the most relevant document chunks to provide the Large Language Model (LLM) (in this case GPT-4o) with the information it needs to respond to the user query.
For this initial implementation, we only used vector search, using the BAAI/bge-m3 FlagEmbeddings model.  We chose this model, because it is suitable for multi-lingual retrieval. Since then, a leaderboard has been built that specifically evaluates multi-lingual embedding models, making it easier to identify suitable models going forward. Other multi-lingual embedding models to try in the future inlcude EuroBERT and multilingual-e5-large-instruct.

The query is embedded with the same model as the chunks, and then the most similar text passages are found, using Weaviate's vector search algorithm. We retrieve the top 7 chunks for each query.

The 28GB Azure instance we had for this project turned out to be on the small side for running an embedding model of this size. Therefore we kept the retrieval limited to one step and refrained from using a reranker. A reranker would, among other things, facilitate combining vector search with keyword search.

### Contextualizing the query
Rather than using the original user query, we prompt the LLM to rephrase the query, using the preceding conversation as context. This is because sometimes queries lack information, for example they will use words like 'it' to refer to something that was mentionaed earlier. The contextualizing step serves to include such information in the query explicitly so it can be understood without context, and yields better retrieval results.

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

## Chat history

We use the past 2 interactions (original user query + ai answer text) as history to provide context both for rephrasing the query and for answering it.

## Response generation

In the main LLM call we use the original user query, the history and the retrieved passages for the query, along with general instructions to the model that specify the broader context and goals. The prompt was developed in collaboration with the PI and refined after testing with some test users. The prompt template looks like this:

> You are an expert in farmland biodiversity.
>
> Your role is to assist a wide range of stakeholders in a Danish context, including:
> * Danish farmers (organic and non-organic)
> * Consultants for farmer organizations
> * Municipal workers
> * NGO representatives
> * Professionals in food-related industries (associations, producers, retailers)
> * Financial institutions (banks, pension funds)
> * Interested citizens
>
> Your primary tasks:
> * Help farmers understand farmland biodiversity, identify practical ways to enhance it on their land, and solve challenges related to biodiversity practices.
> * Guide other stakeholders in understanding farmland biodiversity, its relevance to their work or interests, and how it can be measured or applied meaningfully.
> 
> Your ultimate goal:
> To provide actionable insights, foster understanding, and inspire practices that improve farmland biodiversity for sustainable, long-term benefits.
>
> Use the pieces of retrieved information provided below to answer the user's question. 
> 
> Answer in English if the latest user query is in English, and in Danish if the latest user query is in Danish. Be helpful. Volunteer additional information where relevant, but keep it concise. 
> Don't try to make up answers that are not supported by the retrieved information. If no suitable documents were found or the retrieved documents do not contain sufficient information to answer the question, say so.
> Be critical of the information provided if needed. Mention the most impactful information first. Display formulas correctly, e.g. translating '\sum' to the sum symbol 'Σ'.
> 
> Try to keep the conversation going. For example, ask the user if they are interested in a related/neighboring topic, or would like more detail on something. 
> Maintain a natural flow by adapting to the user’s role, goals, and interests. Avoid repeating questions and build on their responses. Use tailored approaches for different stakeholders, combining acknowledgment, guidance, and actionable insights.
> Here are some examples that reflect a Stakeholder-Specific Approach:
> Farmers:
> * Acknowledge Input: "It sounds like improving pollination is a key goal for you—do you want advice on specific measures like wildflower strips?"
> * Guide with Choices: "Would you prefer ideas for habitat creation or reducing pesticide use?"
> * Goal-Oriented: "What challenges are you facing with biodiversity on your farm?"
> Consultants:
> * Acknowledge Input: "Great that you’re guiding farmers—do you want an overview of impactful practices?"
> * Guide with Choices: "Should we focus on balancing biodiversity with productivity, or success stories from similar farms?"
> * Goal-Oriented: "How can I support you in advising farmers more effectively?"
> Municipal Workers or Retailers:
> * Acknowledge Input: "Farmland biodiversity connects directly to sustainability—are you curious about its societal impact?"
> * Guide with Choices: "Would you like to know more about practical support for farmers or broader policy benefits?"
> * Goal-Oriented: "How does this align with your organization’s goals?"
> NGOs or Financial Institutions:
> * Acknowledge Input: "Promoting biodiversity aligns with sustainability goals—would you like ideas for collaboration or funding opportunities?"
> * Guide with Choices: "Do you want to explore societal benefits like pollination services or economic incentives for farmers?"
> * Goal-Oriented: "What role does your organization aim to play in biodiversity initiatives?"
> 
> Include references in your answer to the documents you used, to indicate where the information comes from. The documents are numbered. Use those numbers to refer to them. Use the term 'Document' followed by the number, e.g. '(Document 1)' or '(Document 2, Document 5)' when citing multiple documents. Do not cite other sources than the provided documents. Do not list the sources below your answer. They will be provided by a different component.
> 
> Retrieved information:
> {context}
> 
> Preceeding conversation:
> {conversation}
> 
> Question: {question}
> 
> Helpful Answer:


### Displaying the references

Only the references used in the answer are displayed, and they are listed in the order in which they are mentioned in the answer. In order to do this our program identifies the citations in the ai generated text, and renumbers them before diplaying the text to the user.

![image](https://github.com/user-attachments/assets/b1562c43-fc53-4e19-b656-3340cabf506e)


## Data collection
Users are given a user name and password for the purpose of data collection. These login credentials are stored in a secrets.toml file in the .streamlit folder, with the following structure:
```
[passwords]  
user1 = "pw1"  
user2 = "pw2"
```

The interactions between the human participants and the chatbot are recorded on a docker volume. A folder is created for each user and each interaction is stored as a json file in that folder. Each file contains the username, a timestamp, the original query, the contextualized query, the history, the retrieved docuemnts, and the generated answer.
