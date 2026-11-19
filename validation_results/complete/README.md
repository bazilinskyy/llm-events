# Completed human validation (frozen)

These are the frozen human validation results for the paper. Validation set ID: `6d23c2baa7387f94` (100 reports).
Both reviewers coded the same 100 reports in the same presentation order.

| File | Contents |
| --- | --- |
| `human_annotations.csv` | One record per reviewer and report (`reviewer1`, `reviewer2`) |
| `human_field_evidence.csv` | The same coding in long format |
| `reviewer_orders.csv` | Presentation order for each reviewer |
| `interrater_agreement.csv` | Reviewer 1 vs Reviewer 2 exact agreement and Cohen's kappa |
| `human_disagreements.csv` | Report and field combinations on which the reviewers disagree |
| `llm_vs_humans.csv` | The LLM compared separately with each reviewer |
| `source_presence_vs_llm.csv` | Human source-presence coding compared with LLM extraction |
| `validation_sample_manifest.csv`, `validation_100.csv`, `validation_set_id.txt` | Sample definition |

The four comparison files were produced with the export functions in `human_validation.py`. `analysis.py` reads
this folder and writes further outputs, including sampling-weighted estimates, to `validation_results/analysis/`.
The supplementary tables cited in the paper are in `validation_results/supplementary/`.

## Correction (November 2026)

The Reviewer 2 coding first reported was not the second reviewer's independent coding, and it has been removed.
`reviewer2` in these files is the second reviewer's actual coding, which the original export labelled `reviewer3`.
The Reviewer 1 coding is unchanged.

Do not edit these files. Store any adjudicated reference separately, for example under
`validation_results/adjudicated/`.
