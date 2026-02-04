# 🩺 Healthcare Document RAG Assistant

A domain-grounded Healthcare Question Answering assistant built using a Retrieval-Augmented Generation (RAG) pipeline.  
The system answers medical questions strictly from approved healthcare documents and provides source citations for every answer.

This project demonstrates an Agentic RAG workflow combining document retrieval, semantic search, and controlled LLM generation.

---

## 🎯 Objective

Large language models can hallucinate in medical contexts. This project solves that by:

- Retrieving relevant medical document passages
- Generating answers only from retrieved context
- Showing document source attribution
- Refusing out-of-scope queries

The result is a grounded, auditable healthcare knowledge assistant.

---

## 🧠 System Architecture

