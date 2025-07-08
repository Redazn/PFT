%% Fusion Full Machine v2 Architecture Diagram
%% For GitHub README.md or documentation

flowchart TD
    classDef input fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef core fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef output fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef special fill:#fce4ec,stroke:#e91e63,stroke-width:2px

    subgraph INPUT_LAYER[Input Layer]
        A[User Query]:::input
        B[Web Scraping\nDuckDuckGo]:::input
        C[Fact Extraction\nGemini AI]:::input
        D[Domain Knowledge]:::input
    end

    subgraph CORE[Processing Core]
        P[Parser\nModal Logic → AST]:::core
        E[Evaluator\nProbabilistic Temporal Reasoning]:::core
        L[Ledger Engine\nBlocks with PoW]:::core
    end

    subgraph OUTPUT[Output Layer]
        R[Results]:::output
    end

    %% Data Flow
    A --> P
    B --> C --> P
    D --> P
    P --> E --> L --> R

    %% Special Components
    subgraph SPECIAL[Key Features]
        PARSER_FEATURES["
            • Belief/Knowledge (Bel, Know)
            • Temporal ops (□/◇)
            • Quantifiers (∀/∃)
            • Predicate logic
        "]:::special

        EVAL_FEATURES["
            • Probabilistic truth (0-1)
            • Entropy measures
            • 80% belief discount
            • Domain-aware quant
        "]:::special

        LEDGER_FEATURES["
            • Blockchain structure
            • Temporal traces
            • Genesis axioms
            • Rule chaining
        "]:::special
    end

    %% Legend
    legend["
        <b>Color Legend:</b>
        <span style='color:#2196f3'>■</span> Input Layer | 
        <span style='color:#4caf50'>■</span> Processing Core | 
        <span style='color:#ff9800'>■</span> Output | 
        <span style='color:#e91e63'>■</span> Features
    "]