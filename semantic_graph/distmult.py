from torch import nn
import torch.nn.functional as F
import torch


class DistMult(nn.Module):

    def __init__(self, num_entities, num_relations, embed_dim, ent_ids, rel_ids):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_embeddings=num_entities, embedding_dim=embed_dim)
        self.relation_embeddings = nn.Embedding(num_embeddings=num_relations, embedding_dim=embed_dim)
        self.ent_ids = ent_ids
        self.rel_ids = rel_ids

    def triplets_to_tensors(self, head_ids, rel_ids, tail_ids):
        h = self.entity_embeddings(head_ids)
        r = self.relation_embeddings(rel_ids)
        t = self.entity_embeddings(tail_ids)
        return h, r, t 


    def forward(self, batch):
        heads, relations, tails = self.triplets_to_tensors(*batch)
        scores = torch.sum(heads * relations * tails, dim=1)
        return F.softmax(scores)


class DistMultSampleGenerator:
    """
    We are doing a couple things hear. Give a triple (head, relation, tail) we are going to randomly "mask" either
    the head or tail. Once chosen (say we chose the head), we need to come up with a bunch of negative samples of the form
    (head_prime, relation, tail). But we need to be especially sure that none of the negative samples are actually true. And then,
    also, we need to make sure that the real triple is mixed in randomly among all the others. 
    """

    def __init__(self, triples, entities, batch_size):
        self.true_triples = triples
        self.entities = entities
        self.batch_size = batch_size

    def sample_entities(self, num_ents):
        return torch.index_select(self.entities, 0, torch.randint(0, len(self.entities), shape=(num_ents,)))

    def make_candidates(self, entities, rest_of_triple, guess_head):
        num_rows = entities.shape[0]
        fixed_ents = torch.tile(rest_of_triple, (num_rows, 1))
        if guess_head:
            triple_candidates = torch.hstack(entities.reshape((num_rows, 1)), fixed_ents)
        else:
            triple_candidates = torch.hstack(fixed_ents, entities.reshape((num_rows, 1)))
        return triple_candidates

    def make_candidate(self, rest_of_triple, guess_head):
        ent = self.sample_entities(1)
        if guess_head:
            triple_candidate = torch.hstack((ent, rest_of_triple))
        else:
            triple_candidate = torch.hstack((rest_of_triple, ent))
        return triple_candidate

    def make_fake_triples(self, rest_of_triple, guess_head):
        num_neg = self.batch_size - 1
        entities = self.sample_entities(num_neg)
        triple_candidates = self.make_candidates(entities, rest_of_triple, guess_head)
        for i, cand in enumerate(triple_candidates):
            if cand in self.true_triples:
                while True:
                    new_cand = self.make_candidate(rest_of_triple, guess_head)
                    if new_cand not in self.true_triples:
                        triple_candidates[i] = new_cand
                        break
        return triple_candidates

    def permute_candidates(self, true_triple, candidates):
        candidates = torch.vstack(true_triple, candidates)
        target = torch.zeros(self.batch_size)
        target[0] = 1
        perm = torch.randperm(self.batch_size)
        return torch.index_select(candidates, 1, perm), torch.index_select(target, 0, perm)

    def generate_batch(self, triple):
        pass

