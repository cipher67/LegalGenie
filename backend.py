
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, GraphCypherQAChain


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Initialize graph and LLM
graph = Neo4jGraph()
llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
llm_transformer = LLMGraphTransformer(llm=llm)

# Define prompts
schema_understanding_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
Given the following graph schema:
{schema}

And the user's question:
{question}

1. Analyze the schema and the question.
2. Identify the key entities and relationships that are relevant to the question.
3. Formulate a Cypher query that will extract the most relevant information from the graph.
4. If the question doesn't seem to directly match the schema, try to find related information that might be helpful.
5. Ensure the query is comprehensive and captures all potentially relevant data.
6. IMPORTANT: Use underscores for multi-word labels (e.g., 'Mobile_app' instead of 'Mobile app').

Cypher Query:
"""
)

summarization_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Summarize the following information from a graph database query:

{context}

The original question was: {question}

Focus on extracting all relevant information to answer the question, including relationships between entities and any important details. Even if the information seems partial or indirect, include it in your summary. Provide a comprehensive summary of all potentially relevant points:
"""
)

answer_generation_prompt = PromptTemplate(
    input_variables=["summary", "question"],
    template="""
Based on the following summary of information from a graph database:

{summary}

Please provide a clear, concise, and informative answer to the following question in proper English:

{question}

Your answer should be in the form of a paragraph or two, addressing the question directly and providing relevant details from the summary. If the information is incomplete or indirect, state this clearly and provide the best possible answer with the available information.
"""
)

# Set up chains
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
answer_generation_chain = LLMChain(llm=llm, prompt=answer_generation_prompt)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=schema_understanding_prompt
)

def process_question(question):
    schema = graph.schema
    print(f"Graph Schema: {schema}")  # Print the schema

    try:
        response = chain.invoke({"query": question, "schema": schema})
        print(f"Raw Response: {response}")  # Print the raw response

        if 'intermediate_steps' in response:
            full_context = response['intermediate_steps']
            print(f"Intermediate Steps: {full_context}")  # Print intermediate steps

            if full_context and isinstance(full_context[-1], tuple):
                # Extract the Cypher query
                cypher_query = full_context[-1][1]
                print(f"Original Cypher Query: {cypher_query}")

                # Replace spaces with underscores if necessary
                cypher_query = cypher_query.replace("Mobile app", "Mobile_app")
                print(f"Modified Cypher Query: {cypher_query}")

                # Execute the Cypher query
                try:
                    context = graph.query(cypher_query)
                    print(f"Query Results: {context}")  # Print query results

                    # Summarize the context, passing both context and question
                    summarized_context = summarization_chain.run(context=str(context), question=question)
                    print(f"Summarized Context: {summarized_context}")  # Print summarized context

                    # Generate a human-readable answer based on the summary
                    final_answer = answer_generation_chain.run(summary=summarized_context, question=question)
                    print(f"Final Answer: {final_answer}")  # Print final answer
                    return final_answer
                except Exception as e:
                    print(f"Error executing Cypher query: {str(e)}")
                    return f"An error occurred while executing the Cypher query: {str(e)}"
            else:
                print("No specific information found in the query results.")
                return "I couldn't find specific information to answer this question based on the graph database query results."
        else:
            # If no intermediate steps, use the result directly if available
            if 'result' in response and response['result'] != "I don't know the answer.":
                print(f"Direct Result: {response['result']}")
                return response['result']
            else:
                # If no specific answer, provide a general response based on the schema
                # general_info = f"Based on the available information in the graph, I can tell you about the following entities and relationships: {schema}"
                # print(f"General Info: {general_info}")
                return f"I couldn't find a specific answer to your question about '{question}'"
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return f"An error occurred while processing the question: {str(e)}"
