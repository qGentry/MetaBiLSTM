# wget -P data/ https://github.com/UniversalDependencies/UD_Russian-SynTagRus/raw/master/ru_syntagrus-ud-train-a.conllu
# wget -P data/ https://github.com/UniversalDependencies/UD_Portuguese-Bosque/raw/master/pt_bosque-ud-train.conllu
# Porttinari-base
wget -P data/ https://github.com/huberemanuel/portinari-base/raw/master/porttinari-base-train.conllu
wget -P data/ https://github.com/huberemanuel/portinari-base/raw/master/porttinari-base-test.conllu
# or any other desired embeddings in default format
# wget -P data/ http://vectors.nlpl.eu/repository/20/182.zip
# unzip data/182.zip -d data/embeddings

# NILC embeddings
wget -O data/skip_s300.zip http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip
# Colab
# cp /content/drive/MyDrive/Colab\ Notebooks/resources/skip_s300.zip data/
unzip data/skip_s300.zip -d data/embeddings