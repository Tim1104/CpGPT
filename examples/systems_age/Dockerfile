FROM rocker/r-ver:4.4.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev libxml2-dev git libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*
RUN R -e "options(Ncpus=4); install.packages(c('remotes','qs2','data.table'))"
RUN R -e "options(Ncpus=4); remotes::install_github('MorganLevineLab/prcPhenoAge', upgrade='never'); remotes::install_github('danbelsky/DunedinPoAm38', upgrade='never'); remotes::install_github('HigginsChenLab/methylCIPHER', upgrade='never')"
