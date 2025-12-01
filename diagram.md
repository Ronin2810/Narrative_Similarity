```mermaid
flowchart LR
  A[PDFs by Ticker] --> B[Extract & Clean Text];
  B --> C[Embed Text];
  C --> D[(pgvector DB)];
  D --> E[Per-Period Vectors];
  E --> F[Cosine Similarity S(i j t)];

  G[Daily Prices] --> H[Returns];
  H --> I[Rolling Correlation r(i j t)];

  F --> J[Align (i j t)];
  I --> J;
  J --> K[Compare S vs r];
