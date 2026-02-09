"""
Sleep-Newspapers Text Mining Pipeline
======================================

A modern Python reimplementation of the R-based analysis of US newspaper
discourse about sleep (1983--2017).  Originally developed by Joon Chung
at the University of Miami Miller School of Medicine.

Modules
-------
config           Centralized parameters and paths.
preprocessing    spaCy-based tokenization, lemmatization, DTM construction.
topic_modeling   LDA (gensim) and BERTopic model training and inspection.
frame_analysis   Dictionary-based media frame scoring and temporal trends.
network_analysis Term co-occurrence network construction (NetworkX).
visualization    Static (matplotlib) and interactive (plotly) figures.
pipeline         CLI orchestrator that ties all modules together.
"""
