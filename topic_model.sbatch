#!/bin/bash
#SBATCH --job-name=topic_model
#SBATCH --output=/home/users/%u/out/topic_model.%j.out
#SBATCH --error=/home/users/%u/err/topic_model.%j.err
#SBATCH -p hns
#SBATCH -c 4
#SBATCH --mem=16GB

module load python/3.9.0
pip3 install nltk gensim spacy
python3 -m spacy download en
python3 arcGeocode.py 
