from setuptools import setup, find_packages

setup(
    name='web_agent_site',
    version="0.1",
    packages=["web_agent_site"],    
    install_requires=[
        'gdown',
        'cleantext==1.1.4',
        'rank_bm25==0.2.2',
        'Flask==2.1.2',
        'Werkzeug==2.3.6',
        'rich==12.4.4',
        'pyserini==0.17.0',
        'faiss-cpu',
        'selenium==4.2.0',
        'spacy==3.3.0',
        'numpy==1.22.4',
        'thefuzz==0.19.0',
        'gym==0.24.0'
    ],
    description="",
    author=""
)