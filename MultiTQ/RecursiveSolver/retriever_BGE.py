import os
import csv
import pickle
import time
import json
import faiss
import faiss.contrib.torch_utils
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import argparse
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker

def parse_date(date_str):
    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %Y"
    ]
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt).date()
            return date_obj
        except ValueError:
            pass
    return None

def extract_dates(text):
    doc = nlp(text)
    dates = ""
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates += ent.text + " "
    dates = dates.strip()
    processed_dates = parse_date(dates)
    return processed_dates


class Retrieval_BGE:
    def __init__(self, d, model_name, triple_list, embedding_size=1024, use_gpu=True, gpu_id=0):
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.model = None
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.triple_list = triple_list
        
        if d == 'time':
            self.fact_list = [f'{f[0]} {f[1]} {f[2]} in {f[3]}.' for f in triple_list]
        else:
            self.fact_list = [f'{f[0]} {f[1]} {f[2]} in {f[3]}.' for f in triple_list]

        self.full_time = [triple[3] for triple in triple_list]
        
        self.index = None
    async def load(self):
        #self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=['cuda:1'])
        
        #self.executor = ThreadPoolExecutor()
        
        self.triplet_embeddings = self.model.encode_corpus(
            self.fact_list,
            convert_to_numpy=True,
            batch_size=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        self.triplet_embeddings = self.triplet_embeddings['dense_vecs']
        #self.triplet_embeddings = self.triplet_embeddings / np.linalg.norm(self.triplet_embeddings, axis=1)[:, None]
        self.triplet_embeddings = self.triplet_embeddings.astype(np.float32)
        print("Embedding shape:", self.triplet_embeddings.shape)
        self.dim=self.triplet_embeddings.shape[-1]
        #self.triplet_embeddings = self.model.encode(self.fact_list, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
        #self.triplet_embeddings = self.triplet_embeddings / np.linalg.norm(self.triplet_embeddings, axis=1)[:, None]
        
        self.index = self.build_faiss_index()
        #self.index = faiss.read_index("./indexfull.bin")

        
        if not self.index.is_trained:
            self.index.train(self.triplet_embeddings)
        self.index.add(self.triplet_embeddings)
        path = "./indexfull.bin"
        faiss.write_index(self.index, path)
        

    def build_faiss_index(self, n_clusters=500, nprobe=60):
        quantizer = faiss.IndexFlatIP(self.embedding_size)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        if self.device == 'cuda':
            ngpu = 1
            resources = [faiss.StandardGpuResources() for _ in range(ngpu)]
            vres = faiss.GpuResourcesVector()
            vdev = faiss.Int32Vector()
            for i, res in zip(range(ngpu), resources):
                vdev.push_back(i)
                vres.push_back(res)
            index_gpu = faiss.index_cpu_to_gpu_multiple(vres, vdev, index)          
            return index_gpu
        else:
            return index
    async def get_embedding(self, corpus_list):
        result =  await asyncio.to_thread(
            self.model.encode_queries,
            corpus_list,
            convert_to_numpy=True,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        return result['dense_vecs']

    async def compute_similarity(self, question, n):
        self.question_embedding = await self.get_embedding([question]) 
        distances, corpus_ids = self.index.search(self.question_embedding, n)
        return distances[0], corpus_ids[0]

    async def get_result(self, question, distances, corpus_ids, re_rank=False):
        if re_rank:
            result = await self.re_rank_single_result(question, distances, corpus_ids)
        else:
            result = await self.basic_result(question, distances, corpus_ids)
        return result

    async def re_rank_single_result(self, question, distances, corpus_ids):
        target_time = extract_dates(question)
        time_list = [10 for _ in range(len(self.full_time))]
        
        if target_time and target_time != "None":
            target_time = datetime.strptime(target_time, "%Y-%m-%d")
            self.adjust_time_scores(question, target_time, time_list)
        
        result = {'question': question}
        hits = [{'corpus_id': id, 'score': score, 'final_score': score * 0.4 - time_list[id] * 0.6}
                for id, score in zip(corpus_ids, distances)]
        hits = sorted(hits, key=lambda x: x['final_score'], reverse=True)

        result['scores'] = [str(hit['score']) for hit in hits][:15]
        result['final_score'] = [str(hit['final_score']) for hit in hits][:15]
        result['triple'] = [self.triple_list[hit['corpus_id']] for hit in hits]
        result['fact'] = [self.fact_list[hit['corpus_id']] for hit in hits]
        return result

    def adjust_time_scores(self, question, target_time, time_list):
        for idx, t in enumerate(self.full_time):
            time_difference = target_time - t
            days_difference = time_difference.days
            if 'before' in question:
                if 0 < days_difference < 16:
                    time_list[idx] = days_difference / 15
            elif 'after' in question:
                if -16 < days_difference < 0:
                    time_list[idx] = -days_difference / 15
            elif 'in' in question and days_difference == 0:
                time_list[idx] = 0

    async def basic_result(self, question, distances, corpus_ids):
        result = {'question': question}
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids, distances)]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        result['scores'] = [str(hit['score']) for hit in hits]
        result['triple'] = [self.triple_list[hit['corpus_id']] for hit in hits]
        result['fact'] = [self.fact_list[hit['corpus_id']] for hit in hits]
        return result

    def save_results(self, result_list, output_path):
        with open(output_path, "w", encoding='utf-8') as json_file:
            json.dump(result_list, json_file, indent=4)
