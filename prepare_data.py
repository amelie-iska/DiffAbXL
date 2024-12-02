#!/usr/bin/env python
"""
Data preparation script for DiffAbXL training using Foldseek for structural clustering.
Downloads and processes SAbDab dataset and splits data based on structural similarity.

Author: Assistant
Updated: 2024-11-30
"""

import os
import sys
import time
import argparse
import logging
import pickle
import lmdb
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import torch
from tqdm.auto import tqdm
import urllib.request
import urllib.error
import gzip
import shutil
import concurrent.futures
from datetime import datetime
import random
import subprocess
from Bio.PDB import PDBIO, Select, StructureBuilder
import copy


# Constants
SABDAB_BASE_URL = "https://opig.stats.ox.ac.uk/webapps/abdb"
SABDAB_SUMMARY_URL = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all"
SABDAB_STRUCTURE_URL = f"{SABDAB_BASE_URL}/entries"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
MAX_WORKERS = 4

# HTTP Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

class DataPreparationError(Exception):
    """Custom exception for data preparation errors."""
    pass

def setup_logger(log_dir: str, debug: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'data_prep_{datetime.now():%Y%m%d_%H%M%S}.log')
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

def three_to_one(residue_name: str) -> str:
    """Convert three-letter amino acid code to one-letter code."""
    return protein_letters_3to1.get(residue_name.upper(), 'X')

def download_with_retry(url: str, max_retries: int = MAX_RETRIES) -> bytes:
    """Download with retry logic and proper error handling."""
    last_exception = None
    
    # Create an opener that handles redirects
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    urllib.request.install_opener(opener)
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with opener.open(req) as response:
                return response.read()
                
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests
                wait_time = (attempt + 1) * RETRY_DELAY
                logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif e.code == 308:  # Permanent Redirect
                if 'Location' in e.headers:
                    url = e.headers['Location']
                    logger.info(f"Following redirect to {url}")
                    continue
            last_exception = e
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            last_exception = e
            
    raise DataPreparationError(f"Failed to download after {max_retries} attempts: {last_exception}")

def inspect_structure(pdb_path: str) -> Dict[str, List[str]]:
    """Inspect a PDB file to determine available chains and their types."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        chain_info = {'chains': [], 'ids': []}
        
        for chain in structure[0]:
            chain_info['chains'].append(chain)
            chain_info['ids'].append(chain.id)
            
        logger.debug(f"Found chains in {pdb_path}: {chain_info['ids']}")
        return chain_info
    except Exception as e:
        logger.error(f"Error inspecting {pdb_path}: {e}")
        return None

def download_sabdab_summary(output_file: str):
    """Download the SAbDab summary file."""
    try:
        logger.info("Downloading SAbDab summary...")
        
        content = download_with_retry(SABDAB_SUMMARY_URL)
        
        # Save the content
        with open(output_file, 'wb') as f:
            f.write(content)
        
        # Verify file content
        with open(output_file) as f:
            header = f.readline().strip().split('\t')
            if not all(field in header for field in ['pdb', 'Hchain', 'Lchain']):
                raise DataPreparationError("Downloaded summary file has incorrect format")
                
        logger.info(f"Successfully downloaded summary file to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to download SAbDab summary: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        raise DataPreparationError("Could not download SAbDab summary file")

def download_pdb(pdb_id: str, output_dir: str) -> bool:
    """Download individual PDB file."""
    pdb_id = pdb_id.lower()
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_path):
        return True
        
    try:
        # Try downloading Chothia numbered version first
        url = f"{SABDAB_STRUCTURE_URL}/{pdb_id}/structure/chothia/{pdb_id}.pdb"
        
        content = download_with_retry(url)
        
        # Save the content
        with open(output_path, 'wb') as f:
            f.write(content)
        
        # Verify file contains ATOM records
        with open(output_path) as f:
            file_content = f.read()
            if "ATOM" not in file_content:
                logger.error(f"Downloaded file for {pdb_id} contains no ATOM records")
                os.remove(output_path)
                return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {pdb_id}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

class AntibodyStructureProcessor:
    """
    Process antibody structures into the required format.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters and settings.
    """
    def __init__(self, config):
        self.config = config
        self.parser = PDB.PDBParser(QUIET=True)
        self.max_atoms = config.get('max_num_heavyatoms', 15)
        self.atom_order = ['N', 'CA', 'C', 'O', 'CB']
    
    def process_structure(self, pdb_path: str, heavy_chain: str = None, 
                        light_chain: str = None, antigen_chains: List[str] = None) -> Dict:
        """
        Process a PDB structure into the required format.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file
        heavy_chain : str, optional
            Chain ID for heavy chain
        light_chain : str, optional
            Chain ID for light chain
        antigen_chains : List[str], optional
            List of chain IDs for antigens

        Returns
        -------
        Dict
            Processed structure dictionary
        """
        try:
            # First inspect the structure
            chain_info = inspect_structure(pdb_path)
            if not chain_info:
                raise DataPreparationError("Failed to inspect structure")

            structure = self.parser.get_structure('protein', pdb_path)
            
            struct_dict = {
                'heavy': None,
                'light': None,
                'structure_type': 'A',
                'light_ctype': None,
                'antigen': [],
                'cdr_sequences': {'heavy': {}, 'light': {}},
                'cdr_structures': {'heavy': {}, 'light': {}},
                'cdr_residues': {'heavy': {}, 'light': {}}
            }
            
            chains = structure[0]
            
            # Process heavy chain
            if heavy_chain and heavy_chain in chains:
                logger.debug(f"Processing heavy chain {heavy_chain} in {pdb_path}")
                try:
                    heavy_chain_data = self._process_chain(chains[heavy_chain], 'heavy')
                    struct_dict['heavy'] = heavy_chain_data
                    # Extract CDR sequences and residues
                    cdr_sequences, cdr_residues = self._extract_cdrs(chains[heavy_chain], 'heavy')
                    logger.debug(f"Extracted CDRs for heavy chain: {list(cdr_residues.keys())}")
                    logger.debug(f"Number of CDR-H3 residues: {len(cdr_residues.get('CDR-H3', []))}")
                    struct_dict['cdr_sequences']['heavy'] = cdr_sequences
                    struct_dict['cdr_residues']['heavy'] = cdr_residues
                except Exception as e:
                    logger.error(f"Error processing heavy chain in {pdb_path}: {str(e)}")
                    struct_dict['heavy'] = None


            # Process light chain
            if light_chain and light_chain in chains:
                try:
                    light_chain_data = self._process_chain(chains[light_chain], 'light')
                    struct_dict['light'] = light_chain_data
                    struct_dict['light_ctype'] = self._get_light_chain_type(chains[light_chain])
                    # Extract CDR sequences and residues
                    cdr_sequences, cdr_residues = self._extract_cdrs(chains[light_chain], 'light')
                    struct_dict['cdr_sequences']['light'] = cdr_sequences
                    struct_dict['cdr_residues']['light'] = cdr_residues
                except Exception as e:
                    logger.error(f"Error processing light chain in {pdb_path}: {str(e)}")
                    struct_dict['light'] = None

            # Process antigen chains if needed
            if antigen_chains:
                for chain_id in antigen_chains:
                    if chain_id in chains:
                        try:
                            antigen_data = self._process_chain(chains[chain_id], 'antigen')
                            struct_dict['antigen'].append(antigen_data)
                        except Exception as e:
                            logger.error(f"Error processing antigen chain {chain_id} in {pdb_path}: {str(e)}")
            
            # Return structure only if at least one chain was processed successfully
            if struct_dict['heavy'] is not None or struct_dict['light'] is not None:
                return struct_dict
            else:
                raise DataPreparationError("No antibody chains were processed successfully")
                
        except Exception as e:
            raise DataPreparationError(f"Error processing {pdb_path}: {str(e)}")

    def _process_chain(self, chain, chain_type: str) -> Dict:
        """
        Process a single chain.

        Parameters
        ----------
        chain : Bio.PDB.Chain
            Chain to process
        chain_type : str
            Type of chain ('heavy', 'light', or 'antigen')

        Returns
        -------
        Dict
            Processed chain data
        """
        try:
            residues = [res for res in chain if is_aa(res, standard=True)]
            if not residues:
                raise DataPreparationError(f"No standard amino acids found in chain {chain.id}")

            aa_list = []
            pos_list = []
            mask_list = []
            resseq_list = []
            icode_list = []

            for residue in residues:
                aa = three_to_one(residue.get_resname())
                aa_list.append(self._aa_to_idx(aa))

                pos, mask = self._process_atoms(residue)
                pos_list.append(pos)
                mask_list.append(mask)

                resseq_list.append(residue.id[1])
                icode_list.append(residue.id[2])

            # Create chain dictionary
            chain_dict = {
                'aa': torch.tensor(aa_list, dtype=torch.long),
                'pos_heavyatom': torch.tensor(np.array(pos_list), dtype=torch.float),
                'mask_heavyatom': torch.tensor(np.array(mask_list), dtype=torch.bool),
                'chain_id': [chain.id] * len(aa_list),
                'resseq': resseq_list,
                'icode': icode_list,
                'res_nb': torch.tensor(resseq_list, dtype=torch.long)
            }

            # Initialize CDR locations for antibody chains
            if chain_type in ['heavy', 'light']:
                chain_dict['cdr_locations'] = self._get_cdr_locations(chain_dict, chain_type)

            return chain_dict

        except Exception as e:
            raise DataPreparationError(f"Error processing chain {chain.id}: {str(e)}")

    def _process_atoms(self, residue: PDB.Residue.Residue) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process atoms in a residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue
            Residue to process

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of atomic positions and masks
        """
        pos = np.zeros((self.max_atoms, 3))
        mask = np.zeros(self.max_atoms, dtype=bool)
        
        # Process backbone and CB atoms first
        for idx, atom_name in enumerate(self.atom_order):
            if atom_name in residue:
                pos[idx] = residue[atom_name].get_coord()
                mask[idx] = True
        
        # Process remaining heavy atoms
        idx = len(self.atom_order)
        for atom in residue:
            if idx >= self.max_atoms:
                break
            if atom.name not in self.atom_order and not atom.element == 'H':
                pos[idx] = atom.get_coord()
                mask[idx] = True
                idx += 1
                
        return pos, mask

    def _aa_to_idx(self, aa: str) -> int:
        """Convert amino acid one-letter code to index."""
        aa_dict = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20
        }
        return aa_dict.get(aa, 20)

    def _get_light_chain_type(self, chain: PDB.Chain.Chain) -> str:
        """Determine the light chain type (kappa or lambda)."""
        sequence = ''
        for residue in chain:
            if is_aa(residue, standard=True):
                aa = three_to_one(residue.get_resname())
                sequence += aa

        # Use sequence motifs to determine the light chain type
        kappa_motifs = ['TVS', 'TVT', 'VTV', 'VSV']
        lambda_motifs = ['TIS', 'TIT', 'VTI', 'VSI']
        
        kappa_score = sum(sequence.count(motif) for motif in kappa_motifs)
        lambda_score = sum(sequence.count(motif) for motif in lambda_motifs)
        
        if kappa_score > lambda_score:
            return 'K'
        elif lambda_score > kappa_score:
            return 'L'
        else:
            return 'K'  # Default to kappa if scores are equal

    def _extract_cdrs(self, chain, chain_type: str) -> Tuple[Dict[str, str], Dict[str, List[PDB.Residue.Residue]]]:
        """
        Extract CDR sequences and residues.

        Parameters
        ----------
        chain : Bio.PDB.Chain
            Chain to process
        chain_type : str
            Type of chain ('heavy' or 'light')

        Returns
        -------
        Tuple[Dict[str, str], Dict[str, List[Bio.PDB.Residue]]]
            CDR sequences and residues
        """
        try:
            cdr_sequences = {}
            cdr_residues = {}
            cdr_definitions = self._get_cdr_definitions(chain_type)
            
            # Get residue numbers as a list for easier lookup
            residue_list = [res for res in chain if is_aa(res, standard=True)]
            logger.debug(f"Total residues in chain: {len(residue_list)}")
            
            for cdr_name, (start_resseq, end_resseq) in cdr_definitions.items():
                sequence = ''
                residues = []
                
                # Log the range we're looking for
                logger.debug(f"Looking for {cdr_name} residues between {start_resseq} and {end_resseq}")
                
                # Find residues within CDR range
                for residue in residue_list:
                    resseq = residue.id[1]
                    logger.debug(f"Checking residue {resseq} for {cdr_name}")
                    if start_resseq <= resseq <= end_resseq:
                        try:
                            aa = three_to_one(residue.get_resname())
                            sequence += aa
                            residues.append(residue)
                            logger.debug(f"Added residue {resseq} ({aa}) to {cdr_name}")
                        except Exception as e:
                            logger.warning(f"Could not process residue {residue.get_resname()} in CDR {cdr_name}: {str(e)}")
                            continue
                
                if sequence and residues:  # Only add if we found valid residues
                    cdr_sequences[cdr_name] = sequence
                    cdr_residues[cdr_name] = residues
                    logger.debug(f"Found {cdr_name}: {sequence} ({len(residues)} residues)")
            
            return cdr_sequences, cdr_residues
                
        except Exception as e:
            raise DataPreparationError(f"Error extracting CDRs: {str(e)}")

    def _get_cdr_definitions(self, chain_type: str) -> Dict[str, Tuple[int, int]]:
        """Get CDR definitions based on Chothia numbering scheme."""
        definitions = {}
        if chain_type == 'heavy':
            definitions = {
                'CDR-H1': (26, 32),
                'CDR-H2': (52, 56),
                'CDR-H3': (95, 102)
            }
        elif chain_type == 'light':
            definitions = {
                'CDR-L1': (24, 34),
                'CDR-L2': (50, 56),
                'CDR-L3': (89, 97)
            }
        logger.debug(f"CDR definitions for {chain_type} chain: {definitions}")
        return definitions

    def _get_cdr_locations(self, chain_dict: Dict, chain_type: str) -> torch.Tensor:
        """
        Determine CDR locations based on Chothia numbering.

        Parameters
        ----------
        chain_dict : Dict
            Dictionary containing chain data
        chain_type : str
            Type of chain ('heavy' or 'light')

        Returns
        -------
        torch.Tensor
            Tensor indicating CDR locations
        """
        cdr_locations = torch.zeros_like(chain_dict['res_nb'])
        cdr_definitions = self._get_cdr_definitions(chain_type)
        
        for cdr_num, (start, end) in enumerate(cdr_definitions.values(), start=1):
            mask = (chain_dict['res_nb'] >= start) & (chain_dict['res_nb'] <= end)
            cdr_locations[mask] = cdr_num
            
        return cdr_locations

class DataPreparation:
    """Main class for preparing the DiffAbXL dataset."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = AntibodyStructureProcessor(config)
        
    def prepare_dataset(self, skip_download: bool = False):
        """Main function to prepare the entire dataset."""
        logger.info("Starting dataset preparation...")
        
        # Create directories
        self._create_directories()
        
        # Process SAbDab summary
        summary_df = self._get_sabdab_summary()
        
        # Download PDB files if needed
        if not skip_download:
            self._download_pdb_files(summary_df)
        else:
            logger.info("Skipping PDB downloads...")
        
        # Process structures
        entries_list = self._process_structures(summary_df)
        
        # Perform structural clustering using Foldseek
        self._perform_structural_clustering(entries_list)
        
        logger.info("Dataset preparation completed successfully!")
        
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['pdb_dir'], exist_ok=True)
        os.makedirs(self.config['processed_dir'], exist_ok=True)
        os.makedirs(self.config['foldseek_dir'], exist_ok=True)
        
    def _get_sabdab_summary(self) -> pd.DataFrame:
        """Download and process SAbDab summary file."""
        summary_file = os.path.join(self.config['data_dir'], 'sabdab_summary_all.tsv')
        
        if not os.path.exists(summary_file):
            download_sabdab_summary(summary_file)
        
        # Read and process summary file
        df = pd.read_csv(summary_file, sep='\t', na_values=['NA', '', 'None'])
        
        logger.info(f"Total entries before filtering: {len(df)}")
        # Convert PDB IDs to lowercase
        df['pdb'] = df['pdb'].str.lower()
        df['Hchain'] = df['Hchain'].astype(str)
        df['Lchain'] = df['Lchain'].astype(str)
        df['antigen_chain'] = df['antigen_chain'].astype(str)
        
        # Convert resolution to float, handling non-numeric values
        df['resolution'] = pd.to_numeric(df['resolution'], errors='coerce')
        
        # Split antigen types and filter based on the configuration
        df['antigen_type'] = df['antigen_type'].astype(str)
        df['antigen_types'] = df['antigen_type'].str.split('|')
        filtered_df = df[
            (df['resolution'].notna()) &  # Remove entries without resolution
            (df['resolution'] <= self.config['resolution_threshold']) &
            df['antigen_types'].apply(lambda x: any(ag_type.strip() in self.config['ag_types'] for ag_type in x))
        ]
        
        # Reset index after filtering
        filtered_df = filtered_df.reset_index(drop=True)
        
        logger.info(f"Found {len(filtered_df)} entries after filtering")
        logger.info(f"Antigen types found: {filtered_df['antigen_type'].unique()}")
        
        return filtered_df
    
    def _check_existing_files(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check which PDB files exist and which need downloading."""
        existing = []
        missing = []
        
        for pdb_id in df['pdb'].unique():
            pdb_path = os.path.join(self.config['pdb_dir'], f"{pdb_id.lower()}.pdb")
            if os.path.exists(pdb_path):
                existing.append(pdb_id)
            else:
                missing.append(pdb_id)
                
        return existing, missing
        
    def _download_pdb_files(self, df: pd.DataFrame):
        """Download PDB files using parallel processing."""
        pdb_ids = df['pdb'].unique()
        
        logger.info(f"Downloading {len(pdb_ids)} PDB files...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_pdb, pdb_id, self.config['pdb_dir'])
                for pdb_id in pdb_ids
            ]
            
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Downloading PDB files"
            ):
                pass
                
    def _process_structures(self, df: pd.DataFrame) -> List[Dict]:
        """Process structures and create LMDB database."""
        lmdb_path = os.path.join(self.config['processed_dir'], 'structures.lmdb')
        
        env = lmdb.open(
            lmdb_path,
            map_size=1024*1024*1024*64,  # 64GB
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True
        )
        
        entries_list = []
        
        with env.begin(write=True) as txn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing structures"):
                try:
                    pdb_id = row['pdb'].lower()
                    pdb_file = os.path.join(self.config['pdb_dir'], f"{pdb_id}.pdb")
                    if not os.path.exists(pdb_file):
                        continue
                        
                    logger.info(f"Processing {pdb_id}...")
                    logger.debug(f"Heavy chain: {row['Hchain']}")
                    logger.debug(f"Light chain: {row['Lchain']}")
                    logger.debug(f"Antigen chain: {row.get('antigen_chain', 'Not specified')}")
                    
                    antigen_chains = row.get('antigen_chain', None)
                    if pd.isna(antigen_chains):
                        antigen_chains = None
                    else:
                        antigen_chains = [ch.strip() for ch in antigen_chains.split(',')]
                    
                    struct_dict = self.processor.process_structure(
                        pdb_file,
                        heavy_chain=row['Hchain'],
                        light_chain=row['Lchain'],
                        antigen_chains=antigen_chains
                    )
                    
                    # Save to LMDB
                    txn.put(
                        pdb_id.encode(),
                        pickle.dumps(struct_dict)
                    )
                    
                    # Add to entries list
                    entries_list.append({
                        'id': pdb_id,
                        'entry': {
                            'ag_name': row.get('antigen', 'Unknown'),
                            'resolution': row['resolution'],
                            'method': row['method'],
                            'scfv': row.get('scfv', False),
                            'light_ctype': struct_dict.get('light_ctype', None),
                            'cdr_sequences': struct_dict['cdr_sequences'],
                            'cdr_structures': struct_dict['cdr_structures']
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {row['pdb']}: {e}")
                    continue
                    
        # Save entries list
        entries_file = os.path.join(self.config['processed_dir'], 'entries_list.pkl')
        with open(entries_file, 'wb') as f:
            pickle.dump(entries_list, f)
            
        return entries_list

    def _perform_structural_clustering(self, entries_list):
        """
        Cluster antibody structures based on structural similarity using Foldseek.
        """
        logger.info("Starting structural clustering using Foldseek...")
        
        # Create a directory for Foldseek files
        foldseek_dir = self.config['foldseek_dir']
        os.makedirs(foldseek_dir, exist_ok=True)
        
        # Extract the heavy chain CDR-H3 regions and save as separate PDB files
        cdr_pdb_dir = os.path.join(foldseek_dir, 'cdr_pdbs')
        os.makedirs(cdr_pdb_dir, exist_ok=True)
        
        pdb_id_to_cdr_pdb = {}  # Mapping from PDB ID to extracted CDR PDB file
        
        # CHANGE 1: Updated extraction and verification of CDR-H3 PDB files
        for entry in entries_list:
            pdb_id = entry['id']
            struct = entry['entry']
            cdr_residues = struct.get('cdr_residues', {})
            heavy_cdrs = cdr_residues.get('heavy', {})
            
            # Extract CDR-H3 residues
            cdr_h3_residues = heavy_cdrs.get('CDR-H3')
            if cdr_h3_residues is not None and len(cdr_h3_residues) > 0:
                # Save CDR-H3 as a PDB file
                cdr_pdb_path = os.path.join(cdr_pdb_dir, f"{pdb_id}_cdrh3.pdb")
                self._write_cdr_pdb_residues(cdr_h3_residues, cdr_pdb_path)
                # Verify file was written successfully
                if os.path.exists(cdr_pdb_path) and os.path.getsize(cdr_pdb_path) > 0:
                    pdb_id_to_cdr_pdb[pdb_id] = cdr_pdb_path
                    logger.debug(f"Successfully wrote CDR-H3 PDB for {pdb_id}")
                else:
                    logger.error(f"Failed to write CDR-H3 PDB for {pdb_id}")
            else:
                logger.warning(f"No CDR-H3 residues for {pdb_id}")

        # CHANGE 2: Add verification before running Foldseek
        if len(pdb_id_to_cdr_pdb) == 0:
            raise DataPreparationError("No valid CDR-H3 PDB files were generated")

        # Create Foldseek database from the CDR PDB files
        db_dir = os.path.join(foldseek_dir, 'foldseek_db')
        os.makedirs(db_dir, exist_ok=True)

        # Write list of PDB files
        pdb_list_file = os.path.join(cdr_pdb_dir, 'pdb_list.txt')
        with open(pdb_list_file, 'w') as f:
            for pdb_id, cdr_pdb_path in pdb_id_to_cdr_pdb.items():
                if os.path.exists(cdr_pdb_path):
                    f.write(f"{cdr_pdb_path}\n")

        # Verify pdb_list.txt was written and contains entries
        if not os.path.exists(pdb_list_file) or os.path.getsize(pdb_list_file) == 0:
            raise DataPreparationError("PDB list file is empty or was not created")

        # CHANGE 4: Add logging to track number of files
        logger.info(f"Generated {len(pdb_id_to_cdr_pdb)} CDR-H3 PDB files")
        logger.info(f"Wrote {sum(1 for line in open(pdb_list_file))} PDB paths to {pdb_list_file}")
            
        # Run Foldseek createdb
        createdb_cmd = [
            'foldseek', 'createdb',
            pdb_list_file,
            os.path.join(db_dir, 'antibody_db')
        ]
        self._run_subprocess(createdb_cmd)
        
        # Run Foldseek all-vs-all search
        result_file = os.path.join(db_dir, 'results.m8')
        search_cmd = [
            'foldseek', 'search',
            os.path.join(db_dir, 'antibody_db'),
            os.path.join(db_dir, 'antibody_db'),
            result_file,
            os.path.join(db_dir, 'tmp'),
            '-a'  # Include alignment in output
        ]
        self._run_subprocess(search_cmd)

        
        # Read Foldseek results and build similarity matrix
        similarity_df = pd.read_csv(
            result_file,
            sep='\t',
            header=None,
            names=[
                'query', 'target', 'prob', 'evalue', 'score',
                'aligned_cols', 'identity', 'similarity', 'query_start', 'query_end',
                'target_start', 'target_end', 'alignment'
            ]
        )
        
        # Map PDB IDs from file paths
        similarity_df['query_pdb'] = similarity_df['query'].apply(lambda x: os.path.basename(x).split('_')[0])
        similarity_df['target_pdb'] = similarity_df['target'].apply(lambda x: os.path.basename(x).split('_')[0])
        
        # Build a distance matrix based on Foldseek scores
        pdb_ids = list(pdb_id_to_cdr_pdb.keys())
        pdb_id_to_index = {pdb_id: idx for idx, pdb_id in enumerate(pdb_ids)}
        num_pdbs = len(pdb_ids)
        distance_matrix = np.zeros((num_pdbs, num_pdbs))
        
        for _, row in similarity_df.iterrows():
            query_idx = pdb_id_to_index[row['query_pdb']]
            target_idx = pdb_id_to_index[row['target_pdb']]
            score = row['score']
            # Convert score to distance (you may need to adjust this based on Foldseek output)
            distance = 1.0 / (score + 1e-6)
            distance_matrix[query_idx, target_idx] = distance
            distance_matrix[target_idx, query_idx] = distance  # Symmetric
        
        # Perform clustering using hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        condensed_dist_matrix = squareform(distance_matrix)
        Z = linkage(condensed_dist_matrix, method='average')
        
        # Determine clusters using a distance threshold
        threshold = self.config.get('clustering_threshold', 0.5)  # Adjust as needed
        cluster_labels = fcluster(Z, t=threshold, criterion='distance')
        
        # Map PDB IDs to cluster labels
        id_to_cluster = {pdb_id: cluster_labels[idx] for pdb_id, idx in pdb_id_to_index.items()}
        
        # Assign clusters to data splits
        unique_clusters = np.unique(cluster_labels)
        np.random.seed(self.config.get('seed', 42))
        np.random.shuffle(unique_clusters)
        
        total_clusters = len(unique_clusters)
        train_ratio = self.config.get('train_ratio', 0.8)
        val_ratio = self.config.get('val_ratio', 0.1)
        num_train_clusters = int(total_clusters * train_ratio)
        num_val_clusters = int(total_clusters * val_ratio)
        num_test_clusters = total_clusters - num_train_clusters - num_val_clusters
        
        train_clusters = unique_clusters[:num_train_clusters]
        val_clusters = unique_clusters[num_train_clusters:num_train_clusters + num_val_clusters]
        test_clusters = unique_clusters[num_train_clusters + num_val_clusters:]
        
        # Assign PDB IDs to splits
        pdb_to_split = {}
        for pdb_id in pdb_ids:
            cluster_label = id_to_cluster[pdb_id]
            if cluster_label in train_clusters:
                pdb_to_split[pdb_id] = 'train'
            elif cluster_label in val_clusters:
                pdb_to_split[pdb_id] = 'val'
            else:
                pdb_to_split[pdb_id] = 'test'
        
        # Write the clustering results
        cluster_file = os.path.join(self.config['processed_dir'], 'cluster_results.tsv')
        with open(cluster_file, 'w') as f:
            for pdb_id, split in pdb_to_split.items():
                f.write(f"{split}\t{pdb_id}\n")
        
        logger.info(f"Cluster file created with splits: {len(train_clusters)} train clusters, {len(val_clusters)} val clusters, {len(test_clusters)} test clusters.")
    
    def _write_cdr_pdb_residues(self, residues: List[PDB.Residue.Residue], output_path: str):
        """
        Write CDR residues to a PDB file.

        Parameters
        ----------
        residues : List[Bio.PDB.Residue.Residue]
            List of residues to write to PDB file
        output_path : str
            Path where to save the PDB file
        """
        try:
            from Bio.PDB import PDBIO
            class CDRSelect(PDB.Select):
                def accept_residue(self, residue):
                    return residue in residues

            io = PDBIO()
            # Create a new structure containing only the selected residues 
            builder = PDB.StructureBuilder.StructureBuilder()
            builder.init_structure('CDR')
            builder.init_model(0)
            builder.init_chain('A')
            
            for residue in residues:
                builder.structure[0]['A'].add(residue)
                
            io.set_structure(builder.structure)
            io.save(output_path, select=CDRSelect())
            
            # Verify file was written
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise DataPreparationError(f"Failed to write PDB file {output_path}")
                
        except Exception as e:
            raise DataPreparationError(f"Error writing PDB file {output_path}: {str(e)}")

    def _write_cdr_pdb(self, coords: np.ndarray, output_path: str):
        """Write CDR coordinates to a PDB file."""
        with open(output_path, 'w') as f:
            for i, coord in enumerate(coords):
                f.write(f"ATOM  {i+1:5d}  CA  ALA A {i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n")
            f.write("END\n")
    
    def _run_subprocess(self, cmd_list: List[str]):
        """Run a subprocess and handle exceptions."""
        try:
            logger.info(f"Running command: {' '.join(cmd_list)}")
            subprocess.run(cmd_list, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error: {e}")
            raise DataPreparationError(f"Subprocess failed: {' '.join(cmd_list)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prepare DiffAbXL training data using Foldseek')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of download workers')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading PDB files')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess existing structures')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    # New flags for clustering parameters
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Ratio of data for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--clustering-threshold', type=float, default=0.5, help='Threshold for clustering')
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Update config with command line arguments
    config.update({
        'data_dir': args.output,
        'pdb_dir': os.path.join(args.output, 'sabdab/pdbs'),
        'processed_dir': os.path.join(args.output, 'processed'),
        'log_dir': os.path.join(args.output, 'logs'),
        'foldseek_dir': os.path.join(args.output, 'foldseek'),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'clustering_threshold': args.clustering_threshold,
    })
    
    # Set up logging
    global logger
    logger = setup_logger(config['log_dir'], debug=args.debug)
    
    try:
        # Set number of workers
        global MAX_WORKERS
        MAX_WORKERS = args.num_workers
        
        # Prepare dataset
        data_prep = DataPreparation(config)
        data_prep.prepare_dataset(skip_download=args.skip_download)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()


# Example usage:
# python prepare_data.py --config config/data_prep.yaml --output data --num-workers 4 --train-ratio 0.8 --val-ratio 0.1 --clustering-threshold 0.35
