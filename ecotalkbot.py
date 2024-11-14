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
import re

#cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2') #sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
#cross_encoder = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)
# @st.cache_resource
# def load_model():
#     return CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

# cross_encoder = load_model()


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



with st.chat_message("ai"):
        st.write("Velkommen til EcoTalkBot – Tal med mig om biodiversitet på landbrugsjord")
        expander = st.expander("Hvad er EcoTalkBot?")
        expander.write("EcoTalkBot er en chatbot udviklet som en del af et forskningsprojekt på Aarhus Universitet, der har til formål at fremme forståelsen og engagementet i biodiversitet på landbrugsjord. Chatbotten giver dig mulighed for at have en interaktiv dialog baseret på pålidelig information i et tilgængeligt format og henviser til de kilder, som svarene bygger på.")
        expander = st.expander("Sådan bruger du EcoTalkBot")
        expander.write("EcoTalkBot er i øjeblikket i en testfase og indsamler data udelukkende til testformål. I denne periode er du velkommen til at interagere med chatbotten ved at stille spørgsmål om biodiversitet på landbrugsjord. Svarene er baseret på udvalgte, pålidelige kilder, som er opført under hvert svar for gennemsigtighed. Testfasen hjælper os med at forbedre chatbotten, samtidig med at vi støtter oplyst dialog om biodiversitet på landbrugsjord.")
        expander = st.expander("Finansiering")
        expander.write("Dette projekt er finansieret af seed funding fra DIGIT, Aarhus University Centre for Digitalisation, Big Data og Data Analytics. EcoTalkBot er en del af EcoMetrics-projektet, som har til formål at udvikle rammer for biodiversitet i landbrugslandskaber. Yderligere finansiering er ydet af Ministeriet for Fødevarer, Landbrug og Fiskeri gennem Organic RRD9, koordineret af ICROFS (Internationalt Center for Forskning i Økologiske Fødevaresystemer) med støtte fra Grønt Udviklings- og Demonstrationsprogram (GUDP). Læs mere om projektet her https://projects.au.dk/sess/projects/ecometric")
        expander = st.expander("Kontakt os")
        expander.write("Hvis du har spørgsmål, er du velkommen til at kontakte den ansvarlige forsker: Gabriele Torma, Sektion for Agricultural Biodiversity, Institut for Agroøkologi, Aarhus Universitet, Email: gtorma@agro.au.dk  \n\n  Tak for din interesse i EcoTalkBot!")
if len(msgs.messages) == 0:
    new_msg = BaseMessage(type='ai', content="Hvordan kan jeg hjælpe dig? / How can I help you?")
    msgs.add_message(new_msg)
    #msgs.add_ai_message("How can I help you?")

template = """You are an expert on biodiversity. Your task is to answer the questions of Danish farmers and consultants of Danish farmer organizations. The goal is  to help farmers get a better understanding of biodiversity, identify opportunities to enhance biodiversity on their land, and solve problems related to biodiversity practices.

Use the following pieces of retrieved information to answer the user's question. 

Answer in English if the latest user query is in English, and in Danish if the latest user query is in Danish. Be helpful. Volunteer additional information where relevant, but keep it concise. 
Don't try to make up answers that are not supported by the retrieved information. If no suitable documents were found or the retrieved documents do not contain sufficient information to answer the question, say so.
Be critical of the information provided if needed. Mention the most impactful information first. Display formulas correctly, e.g. translating '\sum' to the sum symbol 'Σ'.
Try to keep the conversation going. For example, ask the user if they are interested in a related topic, or would like more detail on something.

Include references in your answer to the documents you used, to indicate where the information comes from. The documents are numbered. Use those numbers to refer to them. Use the term 'Document' followed by the number, e.g. '(Document 1)' Do not list the sources below your answer. They will be provided by a different component.

Retrieved information:
{context}

Preceeding conversation:
{conversation}

Question: {question}
Helpful Answer:"""

contextualizing_template = """ Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. The overall topic is biodiversity.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat history:
{history}

Latest user question:
{question}

Standalone version of the question:
"""


def format_docs(docs):
    if docs == []:
        return "No relevant documents were found."
    return"\n\n".join( str(num+1)+') '+doc.page_content+'\n'+json.dumps(doc.metadata, indent=4) for num, doc in enumerate(docs))


#keep sources of previous answers displayed
for msg in msgs.messages:
    if msg.type == "ai" and hasattr(msg, "sources"):
        with st.chat_message("ai"):
            st.write(msg.content)
            expander = st.expander("See sources")
            expander.write(msg.sources)
    else:
        st.chat_message(msg.type).write(msg.content)

#question = st.text_input("Write a question about Gaia: ", key="input")
    
    

############################################
#Refence handling

def add_sources(docs, source_numbers):
    lines = []
    #lines.append('\nSources:')
    if docs and source_numbers:
        for count, num in enumerate(source_numbers):
            rd = docs[int(num)-1]
            doc_info = []
            doc_info.append(str(count+1)+') '+str(rd.metadata["Title"]))
            section_info = []
            for item in rd.metadata:
                if item.startswith('Header'):
                    section_info.append(rd.metadata[item])
            if section_info:
                doc_info.append('  \n   (Section: '+', '.join(section_info)+')')
            doc_info.append('  \n'+rd.metadata["Link"])
            lines.append(''.join(doc_info))
    #text = '\"\"\"'+'\n'.join(lines)+'\"\"\"'
    else:
        lines = ["No relevant information was found in the sources that were selected for this project. Any information provided by the LLM stems from the LLM's internal knowledge and should be checked for accuracy."]
    return '  \n'.join(lines)

def replace_in_text(x, y, text):
    # Define the regex pattern to match 'Document x' with exact match on x
    pattern = rf'Document {x}(?=\b|\D)'
    # Define the replacement text 'Document y'
    replacement = f'Source {y}'
    # Use re.sub() to replace all instances of 'Document x' with 'Document y'
    updated_text = re.sub(pattern, replacement, text) 
    return updated_text

def replace_documents_list(text):
    # Define the regex pattern to match '(Documents x, y, z)'
    pattern = r'\(Documents (\d+(?:, \d+)*(?:,? and \d+)?)\)'
    # Replacement function to reformat the matched text
    def replacement_function(match):
        # Extract the list of numbers from the match
        numbers = match.group(1)
        #print(numbers)
        number_list = re.split(r', and |, | and ', numbers)
        #print(number_list)
        # Join each number with 'Document ' prefix
        new_text = ', '.join([f'Document {num}' for num in number_list])
        #print(new_text)
        # Return the formatted text in the desired format
        return f'({new_text})'
    # Use re.sub() to replace all instances of '(Documents x, y, z)' with '(Document x, Document y, Document z)'
    updated_text = re.sub(pattern, replacement_function, text)
    return updated_text

def f7(seq): #deduplication of list while keeping order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
    
def used_sources(answer):
    listed_pattern = r'\d+, ?\d'
    listed = re.findall(listed_pattern, answer)
    if listed:
        answer = replace_documents_list(answer)
    pattern = r'Document \d+'
    used = re.findall(pattern, answer)
    used = f7(used)
    used = [u.split()[-1] for u in used]
    for num, u in enumerate(used):
        answer = replace_in_text(u, str(num+1), answer)
    return answer, used
#########################################################

if user_input := st.chat_input():
    print(user_input)
    st.chat_message("human").write(user_input)
    prev_conv = '\n'.join([msg.type+': '+msg.content for msg in msgs.messages[-4:]])
    #if len(msgs.messages) > 1:# and contains_referring(user_input):
    contextualizing_prompt = contextualizing_template.format(history=prev_conv, question=user_input)
    print(contextualizing_prompt)
    contextualized_result = gpt3_5.invoke(contextualizing_prompt)
    vector_query = contextualized_result.content
    print(vector_query)
    docs_long = vectorstore.similarity_search(vector_query,k=50)
    farmer_docs = [d for d in docs_long if 'farmer' in d.metadata["Target audience"]]
    docs = farmer_docs[:10]
    if farmer_docs == []:
        docs = docs_long[:7]
    full_prompt = template.format(context=format_docs(docs), question=user_input, conversation=prev_conv)
    print(full_prompt)
    result = gpt4.invoke(full_prompt)
    #sources = add_sources(docs)
    user_msg = BaseMessage(type="human", content=user_input)
    msgs.add_message(user_msg)
    print(result.content)
    ai_answer, source_numbers = used_sources(result.content)
    print(ai_answer)
    sources = add_sources(docs, source_numbers)
    with st.chat_message("ai"):
        st.write(ai_answer)#+add_sources(docs))
        expander = st.expander("See sources")
        expander.write(sources) 
    ai_msg = BaseMessage(type="ai", content=ai_answer)
    setattr(ai_msg, 'sources', sources)
    msgs.add_message(ai_msg)    
    
   
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])