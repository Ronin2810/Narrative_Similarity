<!-- ```mermaid
flowchart LR
  %% ============= STYLES =============
  classDef step fill:#0ea5e9,stroke:#0369a1,color:#fff,stroke-width:1px;
  classDef data fill:#10b981,stroke:#065f46,color:#fff,stroke-width:1px;
  classDef calc fill:#a78bfa,stroke:#5b21b6,color:#fff,stroke-width:1px;
  classDef store fill:#f59e0b,stroke:#b45309,color:#fff,stroke-width:1px;
  classDef out fill:#ef4444,stroke:#7f1d1d,color:#fff,stroke-width:1px;

  %% ============= LEFT BRANCH: TEXT → COSINE =============
  A1["PDFs in data/{TICKER}/*.pdf"]:::data --> A2["Extract text per PDF"]:::step
  A2 --> A3["Clean + Normalize text<br/>(lowercase, whitespace, de-hyphen, drop boilerplate?)"]:::step
  A3 --> A4["Chunk long docs (e.g., 1–2k tokens, 200 overlap)"]:::step
  A4 --> A5["Vectorize chunks → embeddings<br/>(Finance Sentence-Transformer)"]:::calc
  A5 --> A6["Aggregate per document/section<br/>(mean over chunks)"]:::calc
  A6 --> A7["Store in Postgres + pgvector:<br/>document_embeddings(ticker, doc_id, period, section, embedding)"]:::store
  A7 --> A8["Per period: select one vector per firm<br/>(e.g., latest filing or MDA-only)"]:::step
  A8 --> A9["Build all firm pairs within period"]:::step
  A9 --> A10["Cosine similarity Sᵢⱼ,ₜ = vᵢ,ₜ · vⱼ,ₜ<br/>(unit-norm embeddings)"]:::calc
  A10 --> A11["Write similarity_pairs(period, i, j, S, section, model)"]:::store

  %% ============= RIGHT BRANCH: PRICES → CORRELATION =============
  B1["Daily prices per TICKER"]:::data --> B2["Compute daily returns rᵢ,d"]:::step
  B2 --> B3["Map calendar to analysis periods (e.g., quarter-ends)"]:::step
  B3 --> B4["Rolling window per period (e.g., 60d/120d)"]:::step
  B4 --> B5["Pairwise Pearson correlation ρᵢⱼ,ₜ from returns"]:::calc
  B5 --> B6["(Optional) Fisher z: z(ρ)=0.5·ln((1+ρ)/(1-ρ))"]:::calc
  B6 --> B7["Write corr_pairs(period, i, j, ρ or zρ, window)"]:::store

  %% ============= JOIN & COMPARE =============
  A11 --> C1["Align by (period, i, j)"]:::step
  B7  --> C1
  C1 --> C2["Compare Sᵢⱼ,ₜ vs ρᵢⱼ,ₜ (or ρᵢⱼ,ₜ₊₁):<br/>• Regression: z(ρ) ~ S + controls<br/>• Rank/decile checks<br/>• AUC for predicting high future ρ"]:::calc
  C2 --> C3["Report & Visualize:<br/>coeff β, CI, decile curves,<br/>heatmaps (S vs ρ), case timelines"]:::out -->

```mermaid
flowchart LR
  A[PDFs by Ticker] --> B[Extract & Clean Text]
  B --> C[Embed Text]
  C --> D[(pgvector DB)]
  D --> E[Per-Period Firm Vectors]
  E --> F[Cosine Similarity: S(i,j,t)]

  G[Daily Prices] --> H[Returns]
  H --> I[Rolling Corr: ρ(i,j,t)]

  F --> J[Align (i,j,t)]
  I --> J
  J --> K[Compare S vs ρ\n(assoc/predict)]
