from os import environ
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract,metadata,YouTube
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pinecone
from langchain.schema import Document
import streamlit as st

if 'link' not in st.session_state:
    st.session_state.link=''

# function which will fetch the youtube transcript
def download_youtube_transcript(video_url):
    try:
        # Get the transcript for the YouTube video
        video_id=extract.video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine the transcript into a single string
        transcript_text = ""
        for entry in transcript:
            transcript_text += entry['text']
        
        # Write the transcript to a text file
        with open("transcript.txt", "w", encoding="utf-8") as file:
            file.write(transcript_text)
        
    except Exception as e:
        print("An error occurred:", str(e))

    return transcript_text


def main():
    #load environment variables from .env
    load_dotenv()
    openai_api_key=environ.get("OPEN_AI_API_KEY")
    pinecone_api_key=environ.get("PINECONE_API_KEY")
    pinecone_env=environ.get("PINECONE_ENVIRONMENT")
    index_name=environ.get("INDEX_NAME")
    project_name=environ.get("PROJECT_NAME")

    

    embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)

    #initialize pinecone
    pinecone.init(      
	api_key=pinecone_api_key,      
	environment=pinecone_env   
    )      
    # index = pinecone.Index(index_name)
    # indexes = pinecone.list_indexes()
    # if(indexes[0]!="ask-me-anything"):
    #     index=pinecone.create_index(name="ask-me-anything",metric = "cosine",dimension=1500)
    # # this connects to the Pinecone service, creates or retrieves an index 
    # with the given name, vectorizes the input documents, and then inserts them 
    # into the index using the upsert() method.

    docsearch=Pinecone.from_documents([],embedding,index_name=index_name)
            

    #Dividing the transcript into chunks
    st.text_input(label="Youtube Video Url",placeholder="Paste your youtube video link here",on_change=submit,key="widget")
    link=st.session_state.link;
   
    if link and link !='':
        st.session_state.link='';
        thumbnail_url = YouTube(link).thumbnail_url
        video_id = extract.video_id(link)
        #getting the transcript from youtube
        video_transcript=YouTubeTranscriptApi.get_transcript(video_id)
        download_youtube_transcript(link)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        # text will consist of chucks to which the text_splitter divided 
        # texts=text_splitter.split_documents(video_transcript)

        docs=[]
        texts=' '
    
        start=0

        for i in video_transcript:
            if len(texts) >= 2000:
                docs.append(Document(page_content=texts,metadata={"start":start,"youtube_link":link, "thumbnail_url": thumbnail_url}))
                texts=' '       
            else:
                if texts ==' ':
                    start= int(i['start'])
                texts =texts+i['text']

        #storing docs to pinecone
        st.info(len(docs))
        docsearch = Pinecone.from_documents(documents=docs,embedding=embedding,index_name=index_name)
        #resetting link to None to avoid the duplicate insert of docs 
        link=None
        
    user_question = st.text_input("Ask a question about data:")
    if user_question:
    #do the search to DB
        docs = docsearch.similarity_search(user_question)
        llm = OpenAI(openai_api_key=openai_api_key)
         #load question answer from langchain library
        chain = load_qa_chain(llm, chain_type="stuff")
         # get_openai_callback() is used to check the billing info of openai
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        st.write(response)
        references = []
        for j in docs:
            content = j.page_content
            content = (content[: 275] + '-----------') if len(content)>75 else content
            ref_object = {
            'content': content,
            'youtube_link':str(j.metadata['youtube_link']),
            'start':str(int(float(str(j.metadata['start'])))),
            'thumbnail_url':j.metadata['thumbnail_url']
            }
            references.append(ref_object)
        st.header ("References") 
        for j in references:
           st.markdown('<a href="'+str(j['youtube_link' ])+'&t='+str(j['start'])+'">'+str(j['start'])+' <img width="200" height="100" src="'+str(j['thumbnail_url'])+'"> <br>' +j['content'],unsafe_allow_html=True)


# this method called after submit of youtube link
def submit ():
    st.session_state.link = st.session_state.widget
    st.session_state.widget = ""

if __name__ == '__main__':
     main()
    # download_youtube_transcript("https://www.youtube.com/watch?v=CaQA2paqZTE")



