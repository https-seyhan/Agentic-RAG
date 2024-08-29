from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool,QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters,FilterCondition
from typing import List,Optional

import  nest_asyncio
nest_asyncio.apply()

documents = SimpleDirectoryReader(input_files = ['self_rag_arxiv.pdf']).load_data()
print(len(documents))
print(f"Document Metadata: {documents[0].metadata}")

splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Length of nodes : {len(nodes)}")
print(f"get the content for node 0 :{nodes[0].get_content(metadata_mode='all')}")


from llama_index.llms.mistralai import MistralAI
os.environ[""] = userdata.get("")
llm = MistralAI(model="mistral-large-latest")

#instantiate Vectorstore
name = "BERT_arxiv"
vector_index = VectorStoreIndex(nodes,storage_context=storage_context)
vector_index.storage_context.vector_store.persist(persist_path="/content/chroma_db")
#
# Define Vectorstore Autoretrieval tool
def vector_query(query:str,page_numbers:Optional[List[str]]=None)->str:
  '''
  perform vector search over index on
  query(str): query string needs to be embedded
  page_numbers(List[str]): list of page numbers to be retrieved,
                          leave blank if we want to perform a vector search over all pages
  '''
  page_numbers = page_numbers or []
  metadata_dict = [{"key":'page_label',"value":p} for p in page_numbers]
  #
  query_engine = vector_index.as_query_engine(similarity_top_k =2,
                                              filters = MetadataFilters.from_dicts(metadata_dict,
                                                                                    condition=FilterCondition.OR)
                                              )
  #
  response = query_engine.query(query)
  return response
#
#llamiondex FunctionTool wraps any python function we feed it
vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}",
                                              fn=vector_query)
# Prepare Summary Tool
summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                      se_async=True,)
summary_query_tool = QueryEngineTool.from_defaults(name=f"summary_tool_{name}",
                                                    query_engine=summary_query_engine,
                                                  description=("Use ONLY IF you want to get a holistic summary of the documents."
                                              "DO NOT USE if you have specified questions over the documents."))
                                              
 response = llm.predict_and_call([vector_query_tool],
                                "Summarize the content in page number 2",
                                verbose=True)
                                
                                def get_doc_tools(file_path:str,name:str)->str:
  '''
  get vector query and sumnmary query tools from a document
  '''
  #load documents
  documents = SimpleDirectoryReader(input_files = [file_path]).load_data()
  print(f"length of nodes")
  splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
  nodes = splitter.get_nodes_from_documents(documents)
  print(f"Length of nodes : {len(nodes)}")
  #instantiate Vectorstore
  vector_index = VectorStoreIndex(nodes,storage_context=storage_context)
  vector_index.storage_context.vector_store.persist(persist_path="/content/chroma_db")
  #
  # Define Vectorstore Autoretrieval tool
  def vector_query(query:str,page_numbers:Optional[List[str]]=None)->str:
    '''
    perform vector search over index on
    query(str): query string needs to be embedded
    page_numbers(List[str]): list of page numbers to be retrieved,
                            leave blank if we want to perform a vector search over all pages
    '''
    page_numbers = page_numbers or []
    metadata_dict = [{"key":'page_label',"value":p} for p in page_numbers]
    #
    query_engine = vector_index.as_query_engine(similarity_top_k =2,
                                                filters = MetadataFilters.from_dicts(metadata_dict,
                                                                                     condition=FilterCondition.OR)
                                                )
    #
    response = query_engine.query(query)
    return response
  #
  #llamiondex FunctionTool wraps any python function we feed it
  vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}",
                                                fn=vector_query)
  # Prepare Summary Tool
  summary_index = SummaryIndex(nodes)
  summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                       se_async=True,)
  summary_query_tool = QueryEngineTool.from_defaults(name=f"summary_tool_{name}",
                                                     query_engine=summary_query_engine,
                                                    description=("Use ONLY IF you want to get a holistic summary of the documents."
                                                "DO NOT USE if you have specified questions over the documents."))
  return vector_query_tool,summary_query_tool


import os
root_path = "/content/data"
file_name = []
file_path = []
for files in os.listdir(root_path):
  if file.endswith(".pdf"):
    file_name.append(files.split(".")[0])
    file_path.append(os.path.join(root_path,file))
#
print(file_name)
print(file_path)

papers_to_tools_dict = {}
for name,filename in zip(file_name,file_path):
  vector_query_tool,summary_query_tool = get_doc_tools(filename,name)
  papers_to_tools_dict[name] = [vector_query_tool,summary_query_tool]
  

initial_tools = [t for f in file_name for t in papers_to_tools_dict[f]]
initial_tools

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
#
obj_index = ObjectIndex.from_objects(initial_tools,index_cls=VectorStoreIndex)
#

obj_retriever = obj_index.as_retriever(similarity_top_k=2)
tools = obj_retriever.retrieve("compare and contrast the papers self rag and corrective rag")
#
print(tools[0].metadata)
print(tools[1].metadata)

###################################RESPONSE###########################
ToolMetadata(description='Use ONLY IF you want to get a holistic summary of the documents.DO NOT USE if you have specified questions over the documents.', name='summary_tool_self_rag_arxiv', fn_schema=<class 'llama_index.core.tools.types.DefaultToolFnSchema'>, return_direct=False)

ToolMetadata(description='vector_tool_self_rag_arxiv(query: str, page_numbers: Optional[List[str]] = None) -> str\n\n    perform vector search over index on\n    query(str): query string needs to be embedded\n    page_numbers(List[str]): list of page numbers to be retrieved,\n                            leave blank if we want to perform a vector search over all pages\n    ', name='vector_tool_self_rag_arxiv', fn_schema=<class 'pydantic.v1.main.vector_tool_self_rag_arxiv'>, return_direct=False)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
#
agent_worker = FunctionCallingAgentWorker.from_tools(tool_retriever=obj_retriever,
                                                     llm=llm,
                                                     system_prompt="""You are an agent designed to answer queries over a set of given papers.
                                                     Please always use the tools provided to answer a question.Do not rely on prior knowledge.""",
                                                     verbose=True)
agent = AgentRunner(agent_worker)

#
response = agent.query("Compare and contrast self rag and crag.")
print(str(response))

response = agent.query("Summarize the paper corrective RAG.")
print(str(response))
 
