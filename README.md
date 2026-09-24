# Chatbot LLM para Securitização

## 📌 Sobre o projeto

Este projeto tem como objetivo desenvolver um **chatbot especializado em documentos de securitização**, utilizando **Large Language Models (LLMs)** e uma arquitetura de **Retrieval-Augmented Generation (RAG)**.

A proposta é transformar documentos financeiros extensos, como Termos de Securitização e documentos relacionados às operações, em uma base consultável por linguagem natural.

O usuário poderá realizar perguntas sobre uma operação, e o sistema deverá localizar as informações relevantes nos documentos, fornecer o contexto necessário para a LLM e apresentar uma resposta fundamentada nas fontes recuperadas.

---

# 1. Contexto

A securitização é uma atividade do mercado financeiro que envolve a transformação de determinados direitos creditórios ou recebíveis em valores mobiliários que podem ser distribuídos a investidores.

No mercado brasileiro, as operações de securitização podem estar relacionadas a diferentes segmentos econômicos, incluindo os mercados:

- Imobiliário;
- Agropecuário;
- Crédito;
- Precatórios;
- Entre outros.

Esse processo envolve diferentes participantes e estruturas financeiras, nas quais os direitos creditórios são organizados e utilizados como lastro para a emissão de valores mobiliários.

O projeto parte desse contexto para explorar a aplicação de **Inteligência Artificial e Processamento de Linguagem Natural na consulta e interpretação de documentos financeiros**.

---

# 2. Produtos e estruturas de securitização

Entre os instrumentos relacionados ao mercado de recebíveis estão:

| Sigla | Instrumento | Segmento |
|---|---|---|
| CRI | Certificado de Recebíveis Imobiliários | Imobiliário |
| CRA | Certificado de Recebíveis do Agronegócio | Agropecuário |
| DEB | Debêntures | Crédito / Mercado de Capitais |
| CR | Certificados de Recebíveis | Estruturas de recebíveis |

> **Observação:** a estrutura, os participantes, os direitos creditórios e as características dos títulos podem variar de acordo com cada operação.

---

# 3. Objetivo

O objetivo do projeto é desenvolver um **chatbot especializado em securitização**, capaz de consultar documentos de operações financeiras e responder perguntas utilizando as informações presentes nesses documentos.

A aplicação combina:

- LLMs;
- RAG (*Retrieval-Augmented Generation*);
- Embeddings;
- Busca semântica;
- Busca lexical;
- BM25;
- Reranking;
- Processamento e segmentação de documentos;
- APIs;
- Modelos locais;
- Avaliação do sistema.

Em uma etapa futura, o projeto também deverá explorar a disponibilização da aplicação em infraestrutura de nuvem, utilizando **AWS**.

A proposta possui caráter aplicado e busca explorar uma possível utilização comercial da tecnologia dentro do mercado financeiro.

---

# 4. Problema

Documentos de operações de securitização podem possuir centenas de páginas e reunir diferentes tipos de informações:

- características da emissão;
- remuneração;
- prazos;
- participantes;
- garantias;
- fluxo de pagamentos;
- direitos creditórios;
- obrigações contratuais;
- condições da operação;
- anexos e tabelas.

Localizar manualmente uma informação específica pode exigir a leitura de grandes volumes de documentos.

O chatbot busca reduzir esse esforço permitindo que o usuário faça perguntas diretamente em linguagem natural.

### Exemplo

```text
Qual é a remuneração do CRI Meraki?
```
## 5. Avaliação do projeto

A avaliação do projeto busca verificar não apenas se o chatbot consegue gerar uma resposta, mas se ele consegue **encontrar as informações relevantes nos documentos e utilizá-las de forma adequada na geração da resposta**.

Os principais objetivos da avaliação são:

- **Reduzir alucinações:** evitar que a LLM apresente informações que não estejam fundamentadas nos documentos recuperados.
- **Encontrar respostas mais relevantes:** recuperar os trechos do documento que possuem maior relação com a pergunta realizada.
- **Melhorar a precisão do Retrieval:** avaliar a capacidade do sistema de localizar os chunks corretos dentro do documento.
- **Avaliar o ranking dos resultados:** verificar se os trechos relevantes aparecem entre os primeiros resultados recuperados.
- **Avaliar perguntas sem resposta:** verificar se o sistema reconhece quando uma informação não está disponível nos documentos, em vez de inventar uma resposta.
- **Avaliar a qualidade do contexto:** verificar se os trechos recuperados fornecem contexto suficiente para que a LLM responda adequadamente.
- **Avaliar a evolução do RAG:** utilizar métricas de avaliação para acompanhar a evolução do sistema ao longo das diferentes versões do projeto.

### Métricas utilizadas

O projeto utiliza métricas de *Information Retrieval* para avaliar o desempenho da recuperação:

- **Recall@1:** verifica se o chunk esperado aparece em primeiro lugar.
- **Recall@3:** verifica se o chunk esperado aparece entre os três primeiros resultados.
- **Recall@5:** verifica se o chunk esperado aparece entre os cinco primeiros resultados.
- **MRR (Mean Reciprocal Rank):** considera a posição em que o primeiro resultado relevante aparece no ranking.

Além das métricas de *Retrieval*, são realizados testes específicos para **perguntas cuja resposta não está presente nos documentos**, avaliando se o sistema consegue reconhecer a ausência da informação.

> As métricas são utilizadas como indicadores do desempenho do sistema e não representam, isoladamente, a qualidade final das respostas geradas pela LLM.