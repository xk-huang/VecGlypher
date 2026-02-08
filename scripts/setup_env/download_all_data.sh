#!/bin/bash
set -e

# google fonts

mkdir -p data
# commit id: https://github.com/google/fonts/tree/44a3c9a8d8a5b3d6adadedcae000e40e520c55d7

# with-proxy wget https://github.com/google/fonts/archive/44a3c9a8d8a5b3d6adadedcae000e40e520c55d7.zip -O data/google_fonts.zip
# storage_cli mkdirs workspace/data
# storage_cli put data/google_fonts.zip workspace/data/google_fonts.zip --threads 20
storage_cli --prod-use-cython-client get workspace/data/google_fonts.zip data/google_fonts.zip --threads 20
unzip -l data/google_fonts.zip

unzip data/google_fonts.zip -d data
mv data/fonts-main data/google_fonts


# envato fonts

# Download font files
local_envato_base_dir=../envato_fonts/
local_envato_fonts_dir=${local_envato_base_dir}/zip_fonts
local_envato_fonts_tars_dir=${local_envato_base_dir}/zip_fonts_tars
mkdir -p ${local_envato_fonts_dir}
mkdir -p ${local_envato_fonts_tars_dir}

storage_cli --prod-use-cython-client getr --using_direct_reads workspace/data/envato_fonts/zip_fonts_tars ${local_envato_fonts_tars_dir}  --threads 20 --jobs 10
cat ${local_envato_fonts_tars_dir}/fonts.tar.part-* | tar -C ${local_envato_fonts_dir} -xf -

# number of zip files
ls ${local_envato_fonts_dir} | wc -l
# total size of zip files
du -sh ${local_envato_fonts_dir}
# get total number of files after extracting
# python src/tools/check_envato_zips.py ${local_envato_fonts_dir}

# Download metadata
local_envato_base_dir=../envato_fonts
local_envato_metadata_dir=${local_envato_base_dir}/metadata
mkdir -p $local_envato_metadata_dir

storage_metadata_dir=shutterstock_dataset_v2/tree/metadata/fonts/2025-07-31
for metadata_file in fonts_zip_level_metadata.csv fonts_file_level_metadata.csv; do
    storage_cli get ${storage_metadata_dir}/$metadata_file $local_envato_metadata_dir/$metadata_file
done

echo "Done downloading all data"
