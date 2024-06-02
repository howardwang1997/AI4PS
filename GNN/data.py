import joblib
import numpy as np

from typing import Tuple, List
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core import Structure, Molecule

import torch

from graph_utils import laplacian_positional_encoding as lpe
from graph_utils import random_walk_positional_encoding as rwpe
from graph_utils import prepare_line_graph_batch
from graph_utils import compute_bond_cosines, convert_spherical


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, atom_vocab, inputs, outputs, root='molecule_dataset', embedded=False):
        
        if root[-1] != '/':
            root = '%s/' % root
        self.root = root
        self.length = 0
        self.processed = False

        self.inputs = inputs
        self.outputs = outputs
        self.length = len(self.outputs)
        self.dict_graph = {}

        self.atom_vocab = atom_vocab
        self.embedded = embedded
        self.prepare_batch = prepare_line_graph_batch
        
        super().__init__()

    def process(self, name, mol, label):
        g, lg = mol2dglgraph(mol, self.atom_vocab)
        joblib.dump((g, lg, label), '%s/%s.jbl' % (self.root, name))
        return g, lg, label
    
    def __len__(self):
        return self.length
    
    # @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        mol = self.inputs[idx]
        label = self.outputs[idx]
        name = self.outputs.index[idx]

        try:
            item = self.dict_graph[name]
        except KeyError:
            try:
                item = joblib.load('%s/%s.jbl' % (self.root, name))
            except FileNotFoundError:
                item = self.process(name, mol, label)
            self.dict_graph[name] = item

        return item

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)

        return batched_graph, torch.tensor(labels).view(-1,1)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        return batched_graph, batched_line_graph, torch.tensor(labels).view(-1,1)


def mol2dglgraph(smiles, atom_vocab, embedding=False, rdgraph=False, add_h=False, max_nbr=60, max_radius=8):
    """
    dgl graph object
    """
    mol = Chem.MolFromSmiles(smiles)
    if add_h:
        mol = Chem.AddHs(mol)

    if rdgraph:
        raise KeyError('rdgraph not implemented')
    else:
        AllChem.EmbedMolecule(mol)
        pdb_block = Chem.MolToPDBBlock(mol)
        structure = structure_from_pdb(pdb_block)

    if embedding:
        atom_emb = torch.vstack([atom_vocab.get_atom_embedding(structure[i].specie.number)
                                for i in range(len(structure))]) #torch_geometric
    else:
        atom_tokens = [atom.symbol for atom in structure.species]
        atom_emb = torch.Tensor([atom_vocab.vocab.lookup_indices(atom_tokens)]).T
        
    pos = torch.Tensor(structure.cart_coords)
    radius = max_radius
    
    all_nbrs = structure.get_all_neighbors(radius)
    nbr_starts, nbr_ends = [], []
    cart_starts, cart_ends = [], []
    count = 0

    # increment = 0
    # while min([len(nbr) for nbr in all_nbrs]) < max_nbr:
    #     radius += 2
    #     all_nbrs = structure.get_all_neighbors(radius)
    #
    #     increment += 1
    #     print('INCREMENT %d in raius, radius is %d' % (increment, radius))
    
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    for nbr in all_nbrs:
        nbr_ends.extend(list([count]*len(nbr))[:max_nbr])
        nbr_starts.extend(list(map(lambda x: x[2], nbr))[:max_nbr])
        cart_starts.extend(list([pos[count] for _ in range(len(nbr))])[:max_nbr])
        cart_ends.extend(list(map(lambda x: x.coords, nbr))[:max_nbr])

        count += 1
        
    max_idx_atoms = len(atom_emb)-1
    if max_idx_atoms not in nbr_ends:
        nbr_starts.append(max_idx_atoms)
        nbr_ends.append(max_idx_atoms)
        cart_starts.append(pos[max_idx_atoms])
        cart_ends.append(pos[max_idx_atoms])
        print('MODIFIED')
        
    nbr_starts, nbr_ends = np.array(nbr_starts), np.array(nbr_ends)
    
    nbr_starts = torch.Tensor(nbr_starts).long()
    nbr_ends = torch.Tensor(nbr_ends).long()
    
    graph = dgl.graph((nbr_starts, nbr_ends))
    
    graph.edata['r'] = torch.tensor(np.vstack(cart_ends) - np.vstack(cart_starts))

    lg = graph.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    
    graph.ndata['atom_features'] = atom_emb
    graph.edata['spherical'] = convert_spherical(graph.edata['r'])

    graph.ndata['pe'] = torch.cat([lpe(graph, 20), rwpe(graph, 20)], dim=-1)
    graph.edata.pop('r')
    
    line_fea = lg.edata['h']
    lg.edata['h'] = torch.nan_to_num(line_fea, 0.0)
    lg.ndata.pop('r')
    
    return graph, lg


def structure_from_pdb(pdb_block):
    lines = pdb_block.split('\n')
    coords = []
    species = []
    for l in lines:
        if 'REMARKS' in l or 'SITE' in l:
            continue
        if 'ATOM' in l or 'HETATM' in l:
            tmp = l.split()
            c = [float(tmp[5]), float(tmp[6]), float(tmp[7])]
            s = tmp[10]
            coords.append(c)
            species.append(s)
    lattice_mat = np.arrat([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
    structure = Structure(lattice_mat, species, coords, coords_are_cartesian=True)
    return structure
