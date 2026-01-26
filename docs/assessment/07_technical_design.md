# Technical Architecture Proposal

## Overview

This document proposes a phased implementation plan for automating the CSES data harmonization workflow using the specified technology stack.

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | Prefect 3.0 | Workflow management, scheduling, monitoring |
| Agents | LangGraph | Multi-step LLM reasoning, state management |
| LLM Provider | LiteLLM | Unified API for Claude/DeepSeek/GESIS |
| UI | Gradio | Simple web interface for approvals |
| RAG | Haystack | Document retrieval, knowledge base |
| Data Processing | Polars | Fast DataFrame operations |
| Vector DB | Qdrant | Semantic search, embeddings |
| Checkpoints | PostgreSQL | State persistence, audit trail |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Gradio Web UI                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Study Queue │ │ Processing  │ │  Approval   │ │ Dashboard & Monitoring  │ │
│  │   View      │ │   Status    │ │   Queue     │ │                         │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                          Prefect Orchestration                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Study Processing Flow                             │ │
│  │                                                                          │ │
│  │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────────┐ │ │
│  │  │ Setup  │──▶│ Analyze│──▶│Process │──▶│  QA    │──▶│  Finalize      │ │ │
│  │  │ Agent  │   │ Agent  │   │ Agent  │   │ Agent  │   │  Agent         │ │ │
│  │  └────────┘   └────────┘   └────────┘   └────────┘   └────────────────┘ │ │
│  │       │            │            │            │               │          │ │
│  │       ▼            ▼            ▼            ▼               ▼          │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐│ │
│  │  │                    Human Approval Checkpoints                       ││ │
│  │  └─────────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                          LangGraph Agent Layer                               │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────────┐ │
│  │ Document      │ │ Variable      │ │ Code          │ │ Question         │ │
│  │ Analysis Agent│ │ Mapping Agent │ │ Generation    │ │ Generation Agent │ │
│  │               │ │               │ │ Agent         │ │                  │ │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └────────┬─────────┘ │
│          │                 │                 │                  │           │
│          └─────────────────┴────────┬────────┴──────────────────┘           │
│                                     │                                        │
│  ┌──────────────────────────────────▼──────────────────────────────────────┐ │
│  │                         LiteLLM Provider                                 │ │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │ │
│  │   │ Claude API  │  │ DeepSeek R1 │  │ GESIS API   │                     │ │
│  │   │ (Primary)   │  │ (Reasoning) │  │ (Future)    │                     │ │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                          Data & Knowledge Layer                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────────┐ │
│  │  PostgreSQL   │ │    Qdrant     │ │   Haystack    │ │     Polars       │ │
│  │  Checkpoints  │ │  Vector DB    │ │   RAG Index   │ │  Data Processing │ │
│  │  Audit Trail  │ │  Embeddings   │ │  Knowledge    │ │  Transformations │ │
│  └───────────────┘ └───────────────┘ └───────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                          File System (Read-Only Repo)                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  /repo (READ ONLY)          │  /outputs (Generated Files)               │ │
│  │  ├── CSES Data Products/    │  ├── processed_datasets/                  │ │
│  │  ├── CSES Guidelines/       │  ├── generated_code/                      │ │
│  │  └── Templates/             │  └── reports/                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: POC (Weeks 1-2)

**Objective:** Demonstrate core automation capabilities on 1-2 simple studies.

**Deliverables:**
1. Variable mapping agent (Steps 3, 7)
2. Code generation for simple variables
3. Basic Gradio UI for approvals
4. RAG index of workflow.md and knowledge_base.md

**Key Tasks:**
```
Week 1:
├── Set up development environment
├── Create Polars-based data loading
├── Implement frequency generation (Step 6)
├── Build basic RAG pipeline with Haystack
└── Create Gradio prototype

Week 2:
├── Build variable mapping LangGraph agent
├── Implement code generation for F1XXX variables
├── Test on Australia 2022 dataset
└── Document learnings
```

**Success Criteria:**
- Generate 50%+ of F1XXX variable code automatically
- Human approval workflow functional
- RAG retrieves relevant documentation

### Phase 2: Core Pipeline (Weeks 3-5)

**Objective:** Implement Steps 1-7 with full LLM integration.

**Deliverables:**
1. Document analysis agent (Design Report extraction)
2. Variable tracking sheet automation
3. Complete F1XXX-F2XXX code generation
4. Party code mapping agent
5. Prefect workflow orchestration

**Architecture Additions:**
```
├── src/
│   ├── agents/
│   │   ├── document_analyzer.py
│   │   ├── variable_mapper.py
│   │   ├── code_generator.py
│   │   └── party_coder.py
│   ├── workflows/
│   │   ├── setup_flow.py
│   │   ├── analysis_flow.py
│   │   └── processing_flow.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── transformers.py
│   │   └── validators.py
│   └── ui/
│       ├── app.py
│       └── components.py
```

**Key Tasks:**
```
Week 3:
├── Implement document analysis agent
├── Build variable tracking automation
├── Create Prefect flows for Steps 0-3
└── Expand RAG index

Week 4:
├── Implement full code generation agent
├── Add F2XXX (demographics) support
├── Build party code mapping logic
└── PostgreSQL checkpoint integration

Week 5:
├── End-to-end testing on 3 studies
├── UI refinement
├── Error handling improvements
└── Documentation
```

**Success Criteria:**
- Process Steps 1-7 with <20% manual intervention
- 3 completed studies validated against historical outputs
- <5 minute average approval turnaround

### Phase 3: Full Workflow (Weeks 6-8)

**Objective:** Complete all 16 steps with quality assurance.

**Deliverables:**
1. Check file Python port (Step 12)
2. Collaborator question generation (Step 13)
3. ESN and codebook automation (Steps 8, 15)
4. District data agent (Step 9)
5. Complete UI with dashboard

**Key Tasks:**
```
Week 6:
├── Port Stata check files to Python
├── Implement theoretical checks
├── Build validation check logic
└── Integrate with processing flow

Week 7:
├── Implement collaborator question agent
├── Build ESN generation
├── Add district data research agent
└── Enhance approval workflow

Week 8:
├── End-to-end testing on 10 studies
├── Performance optimization
├── Documentation completion
└── User acceptance testing
```

**Success Criteria:**
- Complete 16-step workflow automated
- 70-80% automation rate
- 10 studies processed with <5 days average

### Phase 4: Production (Weeks 9-10)

**Objective:** Production deployment with alternative LLM support.

**Deliverables:**
1. GESIS API integration
2. DeepSeek R1 reasoning mode
3. Production Gradio deployment
4. Complete documentation
5. Training materials

**Key Tasks:**
```
Week 9:
├── GESIS API integration and testing
├── DeepSeek R1 integration for complex reasoning
├── Load testing
└── Security review

Week 10:
├── Production deployment
├── User training
├── Knowledge transfer
└── Handoff documentation
```

**Success Criteria:**
- Production system stable
- Alternative LLM providers functional
- Users trained and productive

## Agent Designs

### Document Analysis Agent (LangGraph)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class DocumentState(TypedDict):
    document_path: str
    document_text: str
    extracted_fields: dict
    confidence_scores: dict
    needs_review: bool

def extract_text(state: DocumentState) -> DocumentState:
    """Extract text from PDF/DOCX"""
    ...

def identify_sections(state: DocumentState) -> DocumentState:
    """Use LLM to identify key sections"""
    ...

def extract_fields(state: DocumentState) -> DocumentState:
    """Extract structured fields using RAG + LLM"""
    ...

def validate_extraction(state: DocumentState) -> str:
    """Route to review or complete"""
    if state["confidence_scores"]["overall"] < 0.8:
        return "needs_review"
    return "complete"

# Build graph
graph = StateGraph(DocumentState)
graph.add_node("extract_text", extract_text)
graph.add_node("identify_sections", identify_sections)
graph.add_node("extract_fields", extract_fields)
graph.add_conditional_edges("extract_fields", validate_extraction, {
    "needs_review": "human_review",
    "complete": END
})
```

### Variable Mapping Agent

```python
class VariableMappingState(TypedDict):
    source_variables: list[dict]
    target_schema: list[dict]
    proposed_mappings: list[dict]
    confidence: float
    unmapped: list[str]

def analyze_source_variables(state: VariableMappingState) -> VariableMappingState:
    """Analyze source variable names, types, and distributions"""
    ...

def retrieve_similar_mappings(state: VariableMappingState) -> VariableMappingState:
    """Use RAG to find similar mappings from past studies"""
    ...

def propose_mappings(state: VariableMappingState) -> VariableMappingState:
    """LLM proposes mappings based on semantics and patterns"""
    ...

def generate_mapping_code(state: VariableMappingState) -> VariableMappingState:
    """Generate Polars transformation code for approved mappings"""
    ...
```

## RAG Knowledge Base Structure

### Document Categories

1. **Workflow Documentation**
   - workflow.md (primary)
   - knowledge_base.md
   - Training materials

2. **Variable Schemas**
   - CSES Module 6 codebook
   - Variable tracking templates
   - Missing value codes

3. **Historical Processing**
   - Past do-files (anonymized patterns)
   - ESN examples
   - Collaborator Q&A patterns

4. **External References**
   - ISCED education mappings
   - Occupation code crosswalks
   - Country code lookups

### Indexing Strategy

```python
from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from qdrant_haystack import QdrantDocumentStore

# Document store configuration
document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="cses_knowledge",
    embedding_dim=1536,
    similarity="cosine"
)

# Indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("splitter", DocumentSplitter(
    split_by="sentence",
    split_length=3,
    split_overlap=1
))
indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
indexing_pipeline.add_component("writer", document_store)
```

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| LLM hallucinations | High confidence thresholds (>0.85), human review for all mappings |
| Stata-Python errors | Validation against 10+ historical outputs, comprehensive test suite |
| GESIS API delays | Design for LiteLLM provider switching, start with Claude API |
| Complex variable failures | Fallback to human mapping, build exception handling |
| User adoption | Simple UI, gradual rollout, extensive training |

## Monitoring & Observability

### Prefect Dashboard Metrics
- Flow execution time
- Task success rates
- Human approval wait times
- Error frequencies

### Custom Metrics
- Variables automated vs manual
- LLM confidence distributions
- Processing time per study
- Accuracy vs historical baseline

## Cost Estimation

### API Costs (per study, estimated)

| Provider | Tokens | Cost |
|----------|--------|------|
| Claude (Sonnet) | ~500K | $1.50-3.00 |
| Claude (Opus) | ~100K | $1.50-2.00 |
| Embeddings | ~200K | $0.02 |
| **Total** | | **$3-5/study** |

### Infrastructure (monthly)

| Component | Cost |
|-----------|------|
| PostgreSQL (managed) | $20-50 |
| Qdrant (self-hosted) | $0 (compute) |
| Gradio (self-hosted) | $0 (compute) |
| **Total** | **$20-50/month** |

## Next Steps

1. **Immediate:** Set up development environment with required packages
2. **Week 1:** Implement POC for Step 6 (frequencies) and basic variable mapping
3. **Ongoing:** Weekly progress reviews, adjust scope as needed
