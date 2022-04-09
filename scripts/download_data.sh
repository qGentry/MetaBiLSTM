# wget -P data/ https://github.com/UniversalDependencies/UD_Russian-SynTagRus/raw/master/ru_syntagrus-ud-train-a.conllu
wget -P data/ https://github.com/UniversalDependencies/UD_Portuguese-Bosque/raw/master/pt_bosque-ud-train.conllu
# or any other desired embeddings in default format
# wget -P data/ http://vectors.nlpl.eu/repository/20/182.zip
# unzip data/182.zip -d data/embeddings
wget -P data/ -O skip_s300.zip http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip
unzip data/skip_s300.zip data/embeddings