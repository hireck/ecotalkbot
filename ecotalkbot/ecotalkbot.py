#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.chains import RetrievalQA
#from langchain.prompts import PromptTemplate
#from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import openai
import hmac
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
#from langchain.chains import LLMChain
from sentence_transformers import CrossEncoder
from langchain_core.messages.base import BaseMessage
#from sentence_transformers import SentenceTransformer
import json

cross_encoder = CrossEncoder('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')



apikey = st.secrets["OPENAIAPIKEY"]
headers = {
    "authorization":apikey,
    "content-type":"application/json"
    }
openai.api_key = apikey



st.title('EcoTalkBot')


@st.cache_resource
def load_vectors():
    embedding_model =  HuggingFaceEmbeddings(model_name='sentence-transformers/distiluse-base-multilingual-cased-v2')#model_name="sentence-transformers/all-MiniLM-L12-v2")#, encode_kwargs={"normalize_embeddings": True},)
    #embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large", encode_kwargs={"normalize_embeddings": True},)#"msmarco-bert-base-dot-v5")
    #embedding_model =  HuggingFaceEmbeddings(model_name="thenlper/gte-large", encode_kwargs={"normalize_embeddings": True}SentenceTransformer('all-MiniLM-L6-v2')
    #embedmodel.max_seq_length = 512
    return FAISS.load_local("faiss_index_v1", embedding_model, allow_dangerous_deserialization=True)

vectorstore = load_vectors()

@st.cache_resource
def load_gpt3_5():
    #return ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    return ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

@st.cache_resource
def load_gpt4():
    return ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    
gpt3_5 = load_gpt3_5()
gpt4 = load_gpt4()
#question = 'Where is the GAIA spacecraft?'

#docs = vectorstore.similarity_search(question,k=5)

msgs = StreamlitChatMessageHistory(key="langchain_messages")
#memory = ConversationBufferMemory(chat_memory=msgs)
#message_history = []


if len(msgs.messages) == 0:
    new_msg = BaseMessage(type='ai', content="How can I help you?")
    msgs.add_message(new_msg)
    #msgs.add_ai_message("How can I help you?")

template = """You are an expert on biodiversity. Your task is to answer the questions of Danish farmers and consultants of Danish farmer organizations. The goal is  to help farmers get a better understanding of biodiversity, identify opportunities to enhance biodiversity on their land, and solve problems related to biodiversity practices.

Use the following pieces of retrieved information to answer the user's question. 

Answer in English if the latest user query is in English, and in Danish if the latest user query is in Danish. Be helpful. Volunteer additional information where relevant, but keep it concise. 
Don't try to make up answers that are not supported by the retrieved information. If the retrieved documents do not contain sufficient information to answer the question, say so.

Include references in your answer to the documents you used, to indicate where the information comes from. The documents are numbered. Use those numbers to refer to them. Do not list the sources below your answer. They will be provided by a different component.

Retrieved information:
{context}

Preceeding conversation:
{conversation}

Question: {question}
Helpful Answer:"""

contextualizing_template = """ Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat history:
{history}

Latest user question:
{question}

Standalone version of the question:
"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#     memory=memory
# )

#prompt = PromptTemplate(input_variables=["context", "question"], template=template)
#llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

#prompt = ChatPromptTemplate.from_template(template)
#chain = prompt | llm



def format_docs(docs):
    return"\n\n".join( str(num+1)+') '+doc.page_content+'\n'+json.dumps(doc.metadata, indent=4) for num, doc in enumerate(docs))


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#    # | StrOutputParser()
# )#


for msg in msgs.messages:
    if msg.type == "ai" and hasattr(msg, "sources"):
        with st.chat_message("ai"):
            st.write(msg.content)
            expander = st.expander("See sources")
            expander.write(msg.sources)
    else:
        st.chat_message(msg.type).write(msg.content)

#question = st.text_input("Write a question about Gaia: ", key="input")
    
    

def add_sources(docs):
    lines = []
    #lines.append('\nSources:')
    for num, rd in enumerate(docs): #result["source_documents"]):
        doc_info = []
        doc_info.append(str(num+1)+') '+str(rd.metadata["Title"]))
        section_info = []
        for item in rd.metadata:
            if item.startswith('Header'):
                section_info.append(rd.metadata[item])
        doc_info.append('   (Section: '+', '.join(section_info)+')')
        doc_info.append('\n'+rd.metadata["Link"])
        lines.append(''.join(doc_info))
    #text = '\"\"\"'+'\n'.join(lines)+'\"\"\"'
    return '\n'.join(lines)


def contains_referring(query):
    words = [t.strip(',.:;?!\"\'') for t in query.split()]
    for w in words:
        if w.lower() in ['the', 'it', 'its', "it's", 'they', 'their', "they're", 'this', 'that' 'these', 'those', 'point', 'item', 'my', 'such', 'here', 'there', 'more', 'most', 'teh']:
            return True
        if w.isdigit():
            return True

if user_input := st.chat_input():
    print(user_input)
    st.chat_message("human").write(user_input)
    prev_conv = '\n'.join([msg.type+': '+msg.content for msg in msgs.messages[-2:]])
    if len(msgs.messages) > 1 and contains_referring(user_input):
        contextualizing_prompt = contextualizing_template.format(history=prev_conv, question=user_input)
        print(contextualizing_prompt)
        contextualized_result = gpt3_5.invoke(contextualizing_prompt)
        vector_query = contextualized_result.content
        print(vector_query)
    else: vector_query = user_input
    retrieved = vectorstore.similarity_search(vector_query,k=20)
    cross_inp = [[vector_query, d.page_content] for d in retrieved]
    cross_scores = cross_encoder.predict(cross_inp)
    scored_pos = [(score, d) for score, d in zip(cross_scores, retrieved) if score > 0]
    if scored_pos:
        reranked = sorted(scored_pos, key=lambda tup: tup[0], reverse=True)
        docs = [r[1] for r in reranked[:7]]
    else:
        scored_neg = [(score, d) for score, d in zip(cross_scores, retrieved)]
        reranked = sorted(scored_neg, key=lambda tup: tup[0], reverse=True) #if there are no positive score, take the 2 least negative ones
        docs = [r[1] for r in reranked[:2]]
    full_prompt = template.format(context=format_docs(docs), question=user_input, conversation=prev_conv)
    print(full_prompt)
    result = gpt4.invoke(full_prompt)
    sources = add_sources(docs)
    user_msg = BaseMessage(type="human", content=user_input)
    msgs.add_message(user_msg)
    with st.chat_message("ai"):
        st.write(result.content)#+add_sources(docs))
        expander = st.expander("See sources")
        expander.write(sources) 
    ai_msg = BaseMessage(type="ai", content=result.content)
    setattr(ai_msg, 'sources', sources)
    msgs.add_message(ai_msg)    
    
   
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])