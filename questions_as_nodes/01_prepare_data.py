"""Step 01 - encode survey answers as 1-5 and build the item table.

In:  ../Survey_Results_UC.csv
Out: outputs/01_responses.csv   respondents x items, 1-5, blank = missing
     outputs/01_items.csv       code, theme, statement text, response counts
     outputs/01_info.json       row counts, missingness
"""
from lib import config
from lib.data import load_responses
from lib.files import save_json


def main() -> None:
    config.ensure_dirs()
    responses, items, info = load_responses(config.DATA_CSV)

    responses.to_csv(config.OUTPUTS / "01_responses.csv")
    items.to_csv(config.OUTPUTS / "01_items.csv", index=False)
    save_json(info, config.OUTPUTS / "01_info.json")

    print(f"Rows in file:        {info['n_rows_in_file']}")
    print(f"Empty rows dropped:  {info['n_empty_rows_dropped']}")
    print(f"Respondents x items: {info['n_respondents']} x {info['n_items']}")
    print(f"Complete cases:      {info['n_complete_cases']}")
    print(f"Missing cells:       {info['n_missing_cells']} "
          f"(of which 'No Comments': {info['n_no_comment_cells']})")


if __name__ == "__main__":
    main()
