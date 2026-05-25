#!/usr/bin/env python3
"""
PRISM Query Examples — Run directly with: python query_examples.py
"""

import os
import sys

try:
    import config
except ImportError:
    config = None

def method_1_agent():
    """Agent (Auto-routing, Recommended)"""
    print("\n" + "="*60)
    print("METHOD 1: Agent (Auto-Routing)")
    print("="*60)
    
    from agents.orchestrator import run_agent
    
    state = run_agent(
        query="What were NVIDIA's key risk factors in 2024?",
        chat_history=[],
        top_k=5
    )
    
    print("\n📌 ANSWER:")
    print(state["answer"])
    
    print("\n📋 SOURCES:")
    for source in state["sources"]:
        print(f"   - {source}")
    
    print(f"\n🔀 ROUTING USED: {state['tool_trace']}")


def method_2_multiturn():
    """Multi-turn conversation with history"""
    print("\n" + "="*60)
    print("METHOD 2: Multi-Turn Conversation")
    print("="*60)
    
    from agents.orchestrator import run_agent
    
    history = []
    
    # Turn 1
    print("\n[Turn 1] What is NVIDIA's data center revenue?")
    result1 = run_agent(
        query="What is NVIDIA's data center revenue?",
        chat_history=history,
        top_k=5
    )
    print(f"Answer: {result1['answer']}\n")
    history.append({"role": "user", "content": "What is NVIDIA's data center revenue?"})
    history.append({"role": "assistant", "content": result1["answer"]})
    
    # Turn 2 (agent sees Turn 1)
    print("[Turn 2] How much did that grow year-over-year?")
    result2 = run_agent(
        query="How much did that grow year-over-year?",
        chat_history=history,
        top_k=5
    )
    print(f"Answer: {result2['answer']}\n")


def method_3_manual_llm():
    """Direct retrieval + manual LLM call"""
    print("\n" + "="*60)
    print("METHOD 3: Manual Retrieval + LLM")
    print("="*60)
    
    from retrieval.hybrid_retriever import HybridRetriever
    from openai import OpenAI
    
    # Step 1: Get chunks
    retriever = HybridRetriever()
    results = retriever.retrieve(
        "What are NVIDIA's main business risks?",
        top_k=5,
        hybrid=True
    )
    
    print(f"\n✓ Retrieved {len(results)} chunks")
    
    # Step 2: Assemble context
    context = "\n\n".join([r.get("text", "") for r in results])
    
    # Step 3: Call LLM
    api_key = (
        (config.LLM_API_KEY if config else "")
        or os.getenv("LLM_API_KEY", "")
        or "not-needed"
    )
    base_url = (
        (config.LLM_BASE_URL if config else "")
        or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    model = (
        (config.LLM_MODEL if config else "")
        or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What are NVIDIA's main business risks?"}
        ],
        temperature=0.2,
        max_tokens=500
    )
    
    print("\n📌 ANSWER:")
    print(response.choices[0].message.content)


def method_4_raw_chunks():
    """Just get chunks, no LLM"""
    print("\n" + "="*60)
    print("METHOD 4: Raw Chunks (No LLM)")
    print("="*60)
    
    from retrieval.hybrid_retriever import HybridRetriever
    
    r = HybridRetriever()
    results = r.retrieve("What were NVIDIA's revenue trends?", top_k=3)
    
    print(f"\n✓ Retrieved {len(results)} chunks:\n")
    for i, chunk in enumerate(results, 1):
        ticker = chunk.get("metadata", {}).get("ticker", "?")
        filing = chunk.get("metadata", {}).get("filing_type", "?")
        text = chunk.get("text", "")[:250]
        
        print(f"[{i}] {ticker} {filing}")
        print(f"    {text}...\n")


def method_5_graph_local():
    """Graph-only local search (entity-focused)"""
    print("\n" + "="*60)
    print("METHOD 5: Graph Local (Entity-Focused)")
    print("="*60)
    
    from retrieval.graph_retriever import GraphRetriever
    
    g = GraphRetriever()
    entity_key = g.resolve_entity_key("NVIDIA")
    if not entity_key:
        entity_key = "nvidia::Company"

    local = g.local_search(entity_keys=[entity_key], top_k=5, hop=2)
    
    print(f"\n✓ Related entities:")
    if "entities" in local:
        for entity in local.get("entities", [])[:5]:
            print(f"   - {entity.get('name', '?')}")
    
    print(f"\n✓ Context:")
    print(local.get("context_text", "")[:500] + "...")
    
    g.close()


def method_6_graph_global():
    """Graph-only global search (thematic)"""
    print("\n" + "="*60)
    print("METHOD 6: Graph Global (Thematic)")
    print("="*60)
    
    from retrieval.graph_retriever import GraphRetriever
    import numpy as np
    
    g = GraphRetriever()
    
    # Use a dummy embedding (in real usage, embed your query)
    embedding = np.random.randn(384).tolist()
    
    global_ctx = g.global_search(
        query_embedding=embedding,
        top_k=3
    )
    
    print(f"\n✓ Top thematic communities:")
    if "communities" in global_ctx:
        for comm in global_ctx.get("communities", [])[:3]:
            print(f"   - {comm.get('title', '?')}")
    
    print(f"\n✓ Context:")
    print(global_ctx.get("context_text", "")[:500] + "...")
    
    g.close()


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║               PRISM Query Examples                             ║
║  Run individual methods: python query_examples.py method1     ║
║  Run all:                python query_examples.py all         ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    examples = {
        "1": ("Agent", method_1_agent),
        "2": ("Multi-Turn", method_2_multiturn),
        "3": ("Manual LLM", method_3_manual_llm),
        "4": ("Raw Chunks", method_4_raw_chunks),
        "5": ("Graph Local", method_5_graph_local),
        "6": ("Graph Global", method_6_graph_global),
    }
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == "all":
            for method_name, method_func in examples.values():
                try:
                    method_func()
                except Exception as e:
                    print(f"\n❌ Error: {e}")
        elif arg in examples:
            try:
                examples[arg][1]()
            except Exception as e:
                print(f"\n❌ Error: {e}")
        else:
            print(f"❌ Unknown: {arg}")
            print(f"Options: {', '.join(examples.keys())}, all")
    else:
        print("\nRun one of:")
        for num, (name, _) in examples.items():
            print(f"  python query_examples.py {num}    # {name}")
        print(f"  python query_examples.py all  # All examples")
