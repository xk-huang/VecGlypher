import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd

output_dir = Path("misc/envato_fonts_stats")
output_dir.mkdir(exist_ok=True, parents=True)

log_file = output_dir / f"{Path(__file__).stem}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    print(f"Using input root: {root}")
else:
    root = Path("data/envato_fonts/zip_fonts")
    print(f"Using default root: {root}")
sub_dir = root.glob("*")
num_dir = 0
num_others = 0
num_zip_files_list = []
num_other_files_list = []
for i in sub_dir:
    if i.is_dir():
        num_dir += 1
        num_zip = 0
        num_others = 0
        for j in i.glob("*"):
            if j.is_dir():
                num_others += 1
            elif j.is_file() and j.suffix == ".zip":
                num_zip += 1
                num_zip_files_list.append(num_zip)
            else:
                num_others += 1
                num_other_files_list.append(num_others)
    else:
        num_others += 1
logger.info(f"num_dir: {num_dir}, num_others: {num_others}")
logger.info(f"num_zip_files_list: {Counter(num_zip_files_list)}")
logger.info(f"num_other_files_list: {Counter(num_other_files_list)}")


plt.hist(num_zip_files_list, bins=100)
plt.xlabel("Number of zip files in each dir")
plt.ylabel("Frequency")
output_fig_path = output_dir / "num_zip_files_hist.png"
plt.savefig(output_fig_path)
print(f"Figure saved to: {output_fig_path}")


def stat_zip(p, return_set=False):
    suffix2set = defaultdict(set)
    try:
        with ZipFile(p) as z:
            pass
    except Exception as e:
        logger.error(f"Failed to open {p}: {e}")
        return None

    with ZipFile(p) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            filename = Path(info.filename)
            # ignore: __MACOSX/*
            if filename.parts[0] == "__MACOSX":
                continue
            # ignore: */.DS_Store
            if filename.stem == ".DS_Store" or filename.suffix == ".DS_Store":
                continue
            if filename.stem == "":
                continue
            if filename.name.startswith("."):
                continue
            suffix = filename.suffix
            suffix2set[suffix].add(filename.stem)
    count_dict = {}
    for suffix, set_ in suffix2set.items():
        count_dict[suffix] = len(set_)
    count_dict["p"] = p
    if return_set:
        return count_dict, suffix2set
    return count_dict


zips = list(root.glob("*/*.zip"))
count_dict_list = []
with ThreadPoolExecutor(max_workers=16) as ex:
    for fut in as_completed([ex.submit(stat_zip, p) for p in zips]):
        result = fut.result()
        if result is None:
            continue
        count_dict_list.append(fut.result())


df = pd.DataFrame(count_dict_list)
# fill NaN with 0
df = df.fillna(0)

df_desc_file = output_dir / "df_desc.csv"
df.describe().to_csv(df_desc_file, sep="\t")
print(f"df_desc saved to: {df_desc_file}")

logger.info(f"ZIPs: {len(df)}")

# # fmt: off
# from IPython import embed; embed()
# # fmt: on
exit()

df[["p", ""]].sort_values(by=[""], ascending=False).iloc[0].to_dict()


from zipfile import ZipFile

zip_file = "data/envato_fonts/f15565fa-76e9-4850-97c1-6d4e62dfa2be/f15565fa-76e9-4850-97c1-6d4e62dfa2be.zip"
with ZipFile(zip_file) as z:
    for info in z.infolist():
        print(
            f"{info.filename} {info.file_size} {info.compress_size} {info.compress_type} {info.is_dir()}"
        )

zip_file = "data/envato_fonts/f15565fa-76e9-4850-97c1-6d4e62dfa2be/f15565fa-76e9-4850-97c1-6d4e62dfa2be.zip"
stat_zip(zip_file, return_set=True)

zip_file = "data/envato_fonts/e4b0d6ad-abbf-46ce-aaab-a13c0437559a/e4b0d6ad-abbf-46ce-aaab-a13c0437559a.zip"
stat_zip(zip_file, return_set=True)

zip_file = "data/envato_fonts/d0c552b5-631c-433c-9cdf-d1b4d1a5513e/d0c552b5-631c-433c-9cdf-d1b4d1a5513e.zip"
stat_zip(zip_file, return_set=True)
