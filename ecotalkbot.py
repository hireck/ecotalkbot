#from langchain.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
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
import datetime
import os
import json

#cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2') #sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
#cross_encoder = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)
# @st.cache_resource
# def load_model():
#     return CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

# cross_encoder = load_model()

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("user not known or password incorrect")
    return False

st.session_state.username = st.session_state.get('username', '') #make sure username is persisted

if not check_password():
    st.stop()

#st.write(f"Welcome, {st.session_state['username']}!")

userdir = '/etc/data/data_collection/'+str(st.session_state['username'])+'/'
if not os.path.isdir(userdir):
    os.makedirs(userdir)

apikey = st.secrets["OPENAIAPIKEY"]
headers = {
    "authorization":apikey,
    "content-type":"application/json"
    }
openai.api_key = apikey



st.title('EcoTalkBot')

with st.sidebar:
    expander = st.expander("Hvad er EcoTalkBot?")
    expander.write("EcoTalkBot er en chatbot udviklet som en del af et forskningsprojekt på Aarhus Universitet, der har til formål at fremme interessen og forståelsen for biodiversitet i landbrugslandet. Gennem en interaktiv dialog med chatbotten har du nem adgang til den nyeste viden omkring biodiversitet. Svarene er baseret et udvalg af pålidelige kilder, som henvises direkte til i svarene. Denne gennemsigtighed er med til at understøtte en let tilgængelig, oplyst dialog omkring biodiversitet i landbruget.")
    expander = st.expander("Sådan bruger du EcoTalkBot")
    expander.write("For at starte en dialog skal du blot skrive dit spørgsmål i feltet nedenfor. EcoTalBot er i øjeblikket i en testfase og indsamler data fra brugerinteraktioner som en del af projektet. Vi bruger udelukkende jeres samtaler til videnskabelige analyser i forbindelse med vores forskning.")
    expander = st.expander("GDPR og datasikkerhed")
    expander.write("Samtaledata og demografiske oplysninger vil blive pseudonymiseret for at sikre GDPR-overholdelse og vil blive behandlet fortroligt. Chatbotten fra universitetet registrerer dine svar, herunder eventuelle personlige oplysninger, som du måtte vælge at dele. For at minimere risikoen opfordrer vi dig til at undgå at inkludere personhenførbare oplysninger i samtalen. Vi opbevarer data sikkert på en krypteret server hos Aarhus Universitet og bruger kun de indsamlede data til videnskabelige, ikke-kommercielle formål, herunder potentielle publikationer og præsentationer. Data fra undersøgelsen opbevares i op til 5 år og slettes derefter sikkert. Alle dataindsamlinger og -behandlinger overholder EU's generelle databeskyttelsesforordning (GDPR) 2016/679. Vi bruger kun dine ikke-følsomme, pseudonymiserede data til forskningsformål. Hvis du har spørgsmål om, hvordan dine data opbevares eller behandles, kan du kontakte Aarhus Universitets databeskyttelsesrådgiver (DPO): Søren Broberg Nielsen via e-mail: soren.broberg@au.dk. Aarhus Universitet, CVR nr. 31119103, er dataansvarlig for behandlingen af dine data.")
    expander = st.expander("Finansiering")
    expander.write("EcoTalkBot er en del af projektet EcoMetric, som har til formål at udvikle rammerne for et biodiversitetsmål, som kan bruges i forvaltningen til at fremme biodiversitet i landbrugslandskaber. EcoTalkBot er finansieret af seed funding fra DIGIT (Centre for Digitalisation, Big Data and Data Analytics), Aarhus Universitet. Yderligere finansiering er ydet af Ministeriet for Fødevarer, Landbrug og Fiskeri gennem Organic RRD9, koordineret af ICROFS (Internationalt Center for Forskning i Økologiske Fødevaresystemer) med støtte fra Grønt Udviklings- og Demonstrationsprogram (GUDP). Læs mere om projektet her https://projects.au.dk/sess/projects/ecometric")
    expander = st.expander("Kontakt os")
    expander.write("Hvis du har spørgsmål, er du velkommen til at kontakte den ansvarlige forsker:  \n\n  Gabriele Torma  \n  Sektion for Agricultural Biodiversity  \n  Institut for Agroøkologi  \n Aarhus Universitet  \n  Email: gtorma@agro.au.dk  \n\n\n  Tak for din interesse i EcoTalkBot!")
    #with st.container:
    st.write("  \n\n  You are logged in as: ")
    #st.write(username)
    st.write(st.session_state["username"])

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
    #return ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, base_url="https://ecotalkbot-ai-service.openai.azure.com/")
    return ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

@st.cache_resource
def load_gpt4():
    #return ChatOpenAI(model_name="gpt-4o", temperature=0)
    #return ChatOpenAI(deployment_name="gpt-4o", temperature=0, openai_api_base="https://au548-m4jm8vwr-swedencentral.cognitiveservices.azure.com/", openai_api_key="4uphKmHYTcaOdGZzb3PQGrSmhPL5Uz1Wtn5xNPpLCbw3k74cqanCJQQJ99ALACfhMk5XJ3w3AAAAACOG9e29")
    return AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-08-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    azure_endpoint="https://au548-m4jm8vwr-swedencentral.cognitiveservices.azure.com",
    # other params...
)
    
#gpt3_5 = load_gpt3_5()
gpt4 = load_gpt4()
#question = 'Where is the GAIA spacecraft?'

#docs = vectorstore.similarity_search(question,k=5)

msgs = StreamlitChatMessageHistory(key="langchain_messages")
#memory = ConversationBufferMemory(chat_memory=msgs)
#message_history = []



#with st.chat_message("ai"):
#    st.write("Velkommen til EcoTalkBot – Tal med mig om biodiversitet på landbrugsjord")


if len(msgs.messages) == 0:
    new_msg = BaseMessage(type='ai', content="Velkommen til EcoTalkBot – Tal med mig om biodiversitet på landbrugsjord  \n\n  Hvordan kan jeg hjælpe dig?")
    msgs.add_message(new_msg)
    #msgs.add_ai_message("How can I help you?")
#st.write(f"Welcome, {st.session_state['username']}!")
template = """You are an expert on biodiversity. Your task is to answer the questions of Danish farmers and consultants of Danish farmer organizations. The goal is  to help farmers get a better understanding of biodiversity, identify opportunities to enhance biodiversity on their land, and solve problems related to biodiversity practices.

Use the following pieces of retrieved information to answer the user's question. 

Answer in English if the latest user query is in English, and in Danish if the latest user query is in Danish. Be helpful. Volunteer additional information where relevant, but keep it concise. 
Don't try to make up answers that are not supported by the retrieved information. If no suitable documents were found or the retrieved documents do not contain sufficient information to answer the question, say so.
Be critical of the information provided if needed. Mention the most impactful information first. Display formulas correctly, e.g. translating '\sum' to the sum symbol 'Σ'.
Try to keep the conversation going. For example, ask the user if they are interested in a related/neighboring topic, or would like more detail on something. For example if they are interested in the lapwing, they may also be interested in other relevant birds, such as the skylark.

Include references in your answer to the documents you used, to indicate where the information comes from. The documents are numbered. Use those numbers to refer to them. Use the term 'Document' followed by the number, e.g. '(Document 1)' or '(Document 2, Document 5)' when citing multiple documents. Do not list the sources below your answer. They will be provided by a different component.

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
        return "Ingen relevante kilder blev fundet."#"No relevant documents were found."
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
        print(source_numbers)
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
        lines = ["De oplysninger, der præsenteres her, refererer ikke eksplicit til de kilder, der blev udvalgt til EcoTalkBot-projektet. Der kan være behov for ekstra forsigtighed med hensyn til nøjagtighed."]
        #lines = ["The information presented here does not explicitly reference the sources that were selected for the EcoTalkBot project. Extra caution with respect to accuracy may be in order."]
    return '  \n'.join(lines)

def replace_in_text(x, y, text):
    # Define the regex pattern to match 'Document x' with exact match on x
    pattern = rf'Document {x}(?=\b|\D)'
    # Define the replacement text 'Document y'
    replacement = f'Kilde {y}'
    # Use re.sub() to replace all instances of 'Document x' with 'Document y'
    updated_text = re.sub(pattern, replacement, text) 
    return updated_text

def replace_documents_list(text):
    # Define the regex pattern to match '(Documents x, y, z)'
    pattern = r'\(Documents? (\d+(?:, \d+)*(?:,? and \d+)?)\)'
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
    
def used_sources(answer, lendocs):
    listed_pattern = r'\d+, ?\d'
    listed = re.findall(listed_pattern, answer)
    if listed:
        answer = replace_documents_list(answer)
    pattern = r'Document \d+'
    used = re.findall(pattern, answer)
    used = f7(used)
    used = [u.split()[-1] for u in used]
    remove = [u for u in used if int(u) > lendocs]
    used = [u for u in used if not u in remove]
    for num, u in enumerate(used):
        answer = replace_in_text(u, str(num+1), answer)
    if remove:
        for rn in remove:
            if '(Document '+rn+')' in answer:
                answer = answer.replace('(Document '+rn+')', '')
            elif 'Document '+rn+',' in answer:
                answer = answer.replace('Document '+rn+', ', '')
            elif ', Document '+rn in answer:
                answer = answer.replace(', Document '+rn, '')
    return answer, used
#########################################################

if user_input := st.chat_input():
    #st.write(f"Welcome, {st.session_state['username']}!")
    print(user_input)
    st.chat_message("human").write(user_input)
    prev_conv = '\n'.join([msg.type+': '+msg.content for msg in msgs.messages[-4:]])
    #if len(msgs.messages) > 1:# and contains_referring(user_input):
    with st.spinner('Henter dokumenter...'):
        contextualizing_prompt = contextualizing_template.format(history=prev_conv, question=user_input)
        print(contextualizing_prompt)
        contextualized_result = gpt4.invoke(contextualizing_prompt)
        vector_query = contextualized_result.content
        print(vector_query)
        docs_long = vectorstore.similarity_search(vector_query,k=50)
        farmer_docs = [d for d in docs_long if 'farmer' in d.metadata["Target audience"]]
        docs = farmer_docs[:10]
        if farmer_docs == []:
            docs = docs_long[:7]
    with st.spinner('Genererer svar...'):
        full_prompt = template.format(context=format_docs(docs), question=user_input, conversation=prev_conv)
        print(full_prompt)
        result = gpt4.invoke(full_prompt)
        user_msg = BaseMessage(type="human", content=user_input)
        msgs.add_message(user_msg)
        print(result.content)
        ai_answer, source_numbers = used_sources(result.content, len(docs))
        print(ai_answer)
        sources = add_sources(docs, source_numbers)
    with st.chat_message("ai"):
        st.write(ai_answer)#+add_sources(docs))
        expander = st.expander("Se kilder")
        expander.write(sources) 
    ai_msg = BaseMessage(type="ai", content=ai_answer)
    setattr(ai_msg, 'sources', sources)
    msgs.add_message(ai_msg)    
    time = datetime.datetime.now()
    interaction = {"user":st.session_state['username'], "date_time":str(time)}
    interaction["user_input"] = user_input
    interaction["contextualized_query"] = vector_query
    interaction["previous_interactions"] = prev_conv
    interaction["retireved_documents"] = []
    for d in docs:
        docjson = {"metadata":d.metadata, "page_content":d.page_content}
        interaction["retireved_documents"].append(docjson)
    interaction["original_answer"] = result.content
    interaction["sources"] = sources
    #for d in sources:
        #docjson = {"metadata":d.metadata, "page_content":d.page_content}
        #interaction["sources"].append(docjson)
    interaction["final_answer"] = ai_answer
    filename = str(time)+'.json'
    path = userdir+filename
    with open(path, 'w') as f:
        json.dump(interaction, f)
    
   
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])