#!/bin/bash
set -e

bash scripts/data_process/250912-build_metadata-google_fonts.sh
bash scripts/data_process/250914-build_metadata-envato_fonts.sh
bash scripts/data_process/250912-build_dataset-google_fonts-alphanumeric.sh
bash scripts/data_process/250912-build_dataset-google_fonts-alphanumeric-abs_coord.sh
bash scripts/data_process/250914-build_dataset-envato_fonts-alphanumeric.sh
bash scripts/data_process/250914-build_dataset-envato_fonts-alphanumeric-abs_coord.sh

echo "Done processing all data"
