"""End-to-End Verification Script for BookMind Phase 5: Conversational RAG API.

Flow:
1. Generate synthetic 2-chapter PDF: "Distributed Architecture & Fault Tolerance"
2. Upload PDF via POST /api/books/upload
3. Index book via POST /api/books/{id}/index
4. Generate Level 2-4 Hierarchies via POST /api/books/{id}/hierarchy/generate
5. Execute live POST /api/books/{id}/chat:
   - Factual inquiry (verifies crux-first answer & structured citations)
   - Multi-turn conversation continuation (verifies local persistence & context)
   - Cross-chapter comparison (verifies dual retrieval & comparative crux)
   - Unanswerable question (verifies insufficient evidence & uncertainty admission)
6. Inspect GET /api/books/{id}/conversations/{conversation_id}
7. Clean up test book.
"""

import json
import os
import sys
import tempfile
import time
import httpx
import pymupdf

BASE_URL = "http://127.0.0.1:8000/api"


def create_test_pdf(filepath: str) -> None:
    """Create a 4-page test PDF with distinct chapters and headings."""
    doc = pymupdf.open()

    # Page 1: Chapter 1 Heading & Quorum Consensus
    p1 = doc.new_page()
    p1.insert_text(
        pymupdf.Point(72, 72),
        "Chapter 1: The Core Principles of Distributed Architecture\n\n"
        "Section 1.1: Quorum Consensus and Leader Election\n"
        "Distributed consensus requires quorum agreement across nodes to prevent split-brain states.\n"
        "In the Raft protocol, nodes transition between follower, candidate, and leader states.\n"
        "A candidate initiates leader election by requesting votes from peers.\n"
        "If a candidate receives votes from a strict majority of nodes, it transitions to leader.\n"
        "Heartbeat messages are periodically broadcast by the leader to suppress new elections.",
        fontsize=11,
    )

    # Page 2: Chapter 1 Section 1.2: Log Replication
    p2 = doc.new_page()
    p2.insert_text(
        pymupdf.Point(72, 72),
        "Chapter 1: The Core Principles of Distributed Architecture\n\n"
        "Section 1.2: Replicated State Machines and Log Matching\n"
        "Log replication ensures that all replicas execute identical state transitions in the same order.\n"
        "When a client sends a command to the leader, the leader appends the entry to its local log.\n"
        "Once a majority of followers acknowledge writing the log entry to persistent storage,\n"
        "the entry is committed and applied to the state machine.",
        fontsize=11,
    )

    # Page 3: Chapter 2 Heading & Partition Tolerance
    p3 = doc.new_page()
    p3.insert_text(
        pymupdf.Point(72, 72),
        "Chapter 2: Fault Tolerance and Partition Handling\n\n"
        "Section 2.1: Network Partitions and Split-Brain Prevention\n"
        "Network partitions sever communication between subsets of nodes in a distributed cluster.\n"
        "Under the CAP theorem, a distributed system facing network partitions must choose between\n"
        "linearizable consistency and uninterrupted availability.\n"
        "To prevent split-brain anomalies, minority partitions must refuse write requests,\n"
        "sacrificing availability to maintain absolute data integrity.",
        fontsize=11,
    )

    # Page 4: Chapter 2 Section 2.2: Byzantine Faults
    p4 = doc.new_page()
    p4.insert_text(
        pymupdf.Point(72, 72),
        "Chapter 2: Fault Tolerance and Partition Handling\n\n"
        "Section 2.2: Byzantine Faults and Asymmetric Communication\n"
        "Crash-stop failures assume nodes either function correctly or cease execution entirely.\n"
        "Byzantine fault tolerance addresses arbitrary or malicious behavior where nodes send conflicting messages.\n"
        "Tolerating f Byzantine nodes requires at least 3f + 1 total nodes in the network,\n"
        "ensuring that honest nodes always command an unassailable majority.",
        fontsize=11,
    )

    doc.save(filepath)
    doc.close()


def run_e2e_verification():
    print("=" * 70)
    print("STARTING BOOKMIND PHASE 5 END-TO-END VERIFICATION")
    print("=" * 70)

    client = httpx.Client(timeout=60.0)

    # Step 0: Check Health
    health_resp = client.get(f"{BASE_URL}/health")
    print(f"[*] Backend Health Check: {health_resp.status_code} - {health_resp.json()}")
    assert health_resp.status_code == 200

    # Step 1: Create PDF
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "distributed_systems_guide.pdf")
    create_test_pdf(pdf_path)
    print(f"[+] Created test PDF at: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    book_id = None
    try:
        # Step 2: Upload PDF
        with open(pdf_path, "rb") as f:
            upload_resp = client.post(
                f"{BASE_URL}/books/upload",
                files={"file": ("distributed_systems_guide.pdf", f, "application/pdf")},
            )
        assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
        book_detail = upload_resp.json()
        book_meta = book_detail["book"]
        book_id = book_meta["book_id"]
        print(f"[+] Uploaded book: ID = {book_id}, Chunks = {book_meta['chunk_count']}, Pages = {book_meta['page_count']}")

        # Step 3: Index Book
        print(f"[*] Indexing book '{book_id}' into Qdrant & BM25...")
        idx_resp = client.post(f"{BASE_URL}/books/{book_id}/index")
        assert idx_resp.status_code == 200, f"Indexing failed: {idx_resp.text}"
        idx_data = idx_resp.json()
        print(f"[+] Indexed successfully: Status = {idx_data['indexing_status']}, Chunks = {idx_data['indexed_chunk_count']}")

        # Step 4: Generate Hierarchical Understanding
        print(f"[*] Synthesizing Level 2-4 Hierarchical Knowledge...")
        hier_resp = client.post(f"{BASE_URL}/books/{book_id}/hierarchy/generate")
        hier_data = hier_resp.json()
        book_level = hier_data.get("book_level")
        thesis = book_level["central_thesis"][:60] if book_level else "N/A"
        print(f"[+] Hierarchy generated: Sections = {len(hier_data['sections'])}, Chapters = {len(hier_data['chapters'])}, Thesis = '{thesis}...'")

        # Step 5: Test Conversational RAG Endpoints
        print("\n" + "=" * 70)
        print("CONVERSATIONAL RAG INQUIRY TESTS (POST /api/books/{id}/chat)")
        print("=" * 70)

        # 5.1 Factual Inquiry (Turn 1)
        print("\n--- Test 5.1: Factual Inquiry (Crux First + Citations) ---")
        q1 = "How does the Raft protocol conduct leader election?"
        payload1 = {"message": q1, "top_k": 3}
        chat_resp1 = client.post(f"{BASE_URL}/books/{book_id}/chat", json=payload1)
        assert chat_resp1.status_code == 200, f"Chat Q1 failed: {chat_resp1.text}"
        d1 = chat_resp1.json()
        conv_id = d1["conversation_id"]

        print(f"Question: {q1}")
        print(f"Question Classification: {d1['question_type']}")
        print(f"Conversation ID: {conv_id}")
        print(f"Crux Summary:\n  {d1['crux_summary']}")
        print(f"Full Answer (Crux-First):\n  {d1['answer']}")
        print(f"Confidence: {d1['confidence']}")
        print(f"Citations Returned ({len(d1['citations'])}):")
        for cit in d1["citations"]:
            print(f"  - Chunk: {cit['chunk_id']} | Pages: {cit['page_start']}-{cit['page_end']} | Chapter: {cit['chapter']}")
            print(f"    Snippet: \"{cit['quoted_snippet'][:80]}...\"")

        assert d1["question_type"] in ("conceptual", "factual")
        assert d1["crux_summary"] != ""
        assert d1["answer"].startswith(d1["crux_summary"])
        assert len(d1["citations"]) >= 1
        assert d1["insufficient_evidence"] is False

        # 5.2 Multi-Turn Follow-up (Turn 2)
        print("\n--- Test 5.2: Multi-turn Follow-up in Existing Conversation ---")
        q2 = "What role do heartbeats play in this process?"
        payload2 = {"conversation_id": conv_id, "message": q2, "top_k": 3}
        chat_resp2 = client.post(f"{BASE_URL}/books/{book_id}/chat", json=payload2)
        assert chat_resp2.status_code == 200, f"Chat Q2 failed: {chat_resp2.text}"
        d2 = chat_resp2.json()
        print(f"Follow-up Question: {q2}")
        print(f"Crux Summary:\n  {d2['crux_summary']}")
        print(f"Citations: {len(d2['citations'])} source chunks cited.")
        assert d2["conversation_id"] == conv_id

        # 5.3 Cross-Chapter Comparison
        print("\n--- Test 5.3: Cross-Chapter Comparison ---")
        q3 = "Compare quorum consensus versus partition tolerance."
        payload3 = {"message": q3, "top_k": 4}
        chat_resp3 = client.post(f"{BASE_URL}/books/{book_id}/chat", json=payload3)
        assert chat_resp3.status_code == 200, f"Chat Q3 failed: {chat_resp3.text}"
        d3 = chat_resp3.json()
        print(f"Comparison Question: {q3}")
        print(f"Classification: {d3['question_type']}")
        print(f"Crux Summary:\n  {d3['crux_summary']}")
        print(f"Synthesized Details:\n  {d3['answer']}")
        assert d3["question_type"] == "cross_chapter_comparison"
        assert len(d3["citations"]) >= 1

        # 5.4 Unanswerable Inquiry (Insufficient Evidence Handling)
        print("\n--- Test 5.4: Unanswerable Inquiry (Insufficient Evidence) ---")
        q4 = "What is the secret recipe for baking a chocolate cake?"
        payload4 = {"message": q4}
        chat_resp4 = client.post(f"{BASE_URL}/books/{book_id}/chat", json=payload4)
        assert chat_resp4.status_code == 200, f"Chat Q4 failed: {chat_resp4.text}"
        d4 = chat_resp4.json()
        print(f"Unanswerable Question: {q4}")
        print(f"Insufficient Evidence Flag: {d4['insufficient_evidence']}")
        print(f"Confidence: {d4['confidence']}")
        print(f"Answer Output:\n  {d4['answer']}")
        print(f"Citations: {d4['citations']}")
        assert d4["insufficient_evidence"] is True
        assert d4["confidence"] == "insufficient"
        assert len(d4["citations"]) == 0

        # Step 6: Inspect Conversation History Thread
        print("\n--- Test 6: Verify Local Conversation Persistence ---")
        conv_resp = client.get(f"{BASE_URL}/books/{book_id}/conversations/{conv_id}")
        assert conv_resp.status_code == 200, f"Fetch conversation failed: {conv_resp.text}"
        conv_obj = conv_resp.json()
        print(f"Conversation ID: {conv_obj['conversation_id']}")
        print(f"Total Messages Logged: {len(conv_obj['messages'])}")
        for idx, m in enumerate(conv_obj["messages"]):
            print(f"  [{idx + 1}] Role: {m['role']} | Length: {len(m['content'])} chars | Citations: {len(m['citations'] or [])}")
        assert len(conv_obj["messages"]) == 4

        print("\n" + "=" * 70)
        print("ALL END-TO-END CHECKS COMPLETED WITH 100% SUCCESS!")
        print("=" * 70)

    finally:
        # Cleanup PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        # Cleanup uploaded test book
        if book_id:
            del_resp = client.delete(f"{BASE_URL}/books/{book_id}")
            print(f"[*] Cleaned up test book {book_id}: {del_resp.status_code}")


if __name__ == "__main__":
    run_e2e_verification()
