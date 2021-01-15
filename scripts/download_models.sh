mkdir -p all_models
wget https://dl.google.com/coral/canned_models/all_models.tar.gz
tar -C all_models -xvzf all_models.tar.gz
rm -f all_models.tar.gz