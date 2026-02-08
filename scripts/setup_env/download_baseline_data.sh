#!/bin/bash
# workspace/data/baseline_data
files=(
deepvecfont_v1-font_ttfs.zip
deepvecfont_v2-data.zip
dualvector-dvf_png.zip
svg_vae-glyphazzn_urls.txt
tag_fonts-dataset.tar.gz
)
storage_dir=workspace/data/baseline_data
local_dir=../baseline_data

mkdir -p $local_dir

for file in "${files[@]}"; do
    local_file=$local_dir/$file
    storage_file=$storage_dir/$file
    echo "Downloading $storage_file to $local_file"
    storage_cli --prod-use-cython-client get --threads 20 $storage_file $local_file
done


for file in "${files[@]}"; do
    # if file is ending with .zip or .tar.gz, or .tar, unzip it
    if ! [[ $file == *.zip || $file == *.tar.gz || $file == *.tar ]]; then
        echo "Skipping $file"
        continue
    fi
    local_file=$local_dir/$file
    local_unzip_dir=$local_dir/$(echo "$file" | sed 's/\.[^.]*\(\.[^.]*\)*$//')
    echo "Unzipping $local_file to $local_unzip_dir"
    if [[ $file == *.zip ]]; then
        unzip -o $local_file -d $local_unzip_dir
    elif [[ $file == *.tar.gz ]]; then
        tar -xzf $local_file -C $local_unzip_dir
    elif [[ $file == *.tar ]]; then
        tar -xf $local_file -C $local_unzip_dir
    fi
done
