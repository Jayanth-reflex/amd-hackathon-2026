# Domain Dataset Sources — curated from deep research (2026-04-26)

> Output of 4 parallel research agents tasked with finding HIGH-QUALITY, license-clean (Apache-2.0 / MIT / CC-BY / PD) sources for our 14 domain sub-areas. The model card cites these by domain.

License rule for inclusion: must be compatible with downstream Apache-2.0 model weights. Excluded: CC-BY-NC*, CC-BY-NC-ND, GFDL-only, AGPL, "research only" without explicit license, NCERT/NPTEL (govt copyright unclear), WHO (CC-BY-NC-SA-3.0 IGO), Hesperian Open License, MIT OCW (CC-BY-NC-SA), Open Yale (CC-BY-NC-SA), Stanford Encyclopedia of Philosophy (CC-BY-NC-ND).

---

## A. Indian Law (target ~5M tokens)

### A.1 HF datasets (Apache-2.0 / MIT / CC-BY)
| HF ID | License | Notes |
|---|---|---|
| `opennyaiorg/aalap_instruction_dataset` | Apache/CC0/MIT (drop CC-BY-NC contract-clause subset) | Best Indian-legal synthetic instruction dataset; 22k Q&A on FIRs, judgments, statute interpretation |
| `opennyaiorg/InJudgements_dataset` | Apache-2.0 | 13.5k IndianKanoon judgments 1950-2017, 8 case types, 70M raw text tokens |
| `viber1/indian-law-dataset` | Apache-2.0 | 24,607 instruction pairs on FIR/bail/writ/plaint procedure |
| `rishiai/indian-court-judgements-and-its-summaries` | Apache-2.0 | 6,944 SC judgments with summaries |
| `nisaar/Lawyer_GPT_India` | Apache-2.0 | Constitution/IPC/Contract Q&A |
| `nisaar/Constitution_of_India` | Apache-2.0 | Article-level Q&A |
| `nisaar/Articles_Constitution_3300_Instruction_Set` | Apache-2.0 | Constitutional reasoning |
| `nisaar/Constitution_Of_India_Instruction_Set` | Apache-2.0 | SC case-reasoning instructions |
| `Sharathhebbar24/Indian-Constitution` | Apache-2.0 | Article-by-article Constitution text |

### A.2 Government PDFs (PD per Indian Copyright Act §52(1)(q))
- BNS 2023: `https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf`
- BNSS 2023: `https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf`
- BSA 2023: `https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf`
- IPC 1860 (legacy): `https://www.indiacode.nic.in/bitstream/123456789/11091/1/the_indian_penal_code,_1860.pdf`
- CrPC 1973 (legacy): `https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure,_1973.pdf`
- Constitution of India: `https://www.indiacode.nic.in/bitstream/123456789/19150/1/constitution_of_india.pdf`
- BNS handbook (BPRD): `https://bprd.nic.in/uploads/pdf/BNS_English_30-04-2024.pdf`

### A.3 Bulk SC/HC judgments (CC-BY-4.0 via AWS Open Data)
- SC judgments: `s3://indian-supreme-court-judgments/` — 52GB, 35k judgments (no-sign-request)
- HC judgments: `s3://indian-high-court-judgments/` — 1TB across 25 high courts (sample selectively)

### A.4 Excluded
- `Exploration-Lab/IL-TUR` — CC-BY-NC-SA-4.0
- `Techmaestro369/indian-legal-texts-finetuning` — CC-BY-SA-4.0 (ShareAlike concerns)
- NCERT — govt copyright unclear

---

## B. Indian Taxation (target ~3M tokens)

### B.1 Government Acts (PD)
- Income Tax Act 1961 (FA 2025): `https://incometaxindia.gov.in/Documents/income-tax-act-1961-as-amended-by-finance-act-2025.pdf`
- Income Tax Act (FA 2026): `https://www.incometaxindia.gov.in/documents/d/guest/income_tax_act_1961_as_amended_by_fa_act_2026-1-pdf`
- CGST Act 2017: `https://cbic-gst.gov.in/pdf/CGST-Act-Updated-31082021.pdf`
- IGST Act 2017: `https://cbic-gst.gov.in/aces/Documents/IGST-bill-e.pdf`
- CGST Rules Part A: `https://cbic-gst.gov.in/pdf/01062021-CGST-Rules-2017-Part-A-Rules.pdf`
- CGST Rules Part B: `https://cbic-gst.gov.in/pdf/01012022-CGST-Rules-2017-amended-Part-B.pdf`
- CBIC GST FAQ 2nd ed: `https://cbic-gst.gov.in/pdf/new-faq-on-gst-second-edition.pdf`
- CBIC GST FAQ original: `https://cbic-gst.gov.in/aces/Documents/faq-on-gst.pdf`
- Companies Act 2013: `https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf`

---

## C. Pharmacology (target ~3M tokens)

### C.1 HF datasets
| HF ID | License | Notes |
|---|---|---|
| `qiaojin/PubMedQA` | MIT | 273k Yes/No/Maybe biomedical QA |
| `openlifescienceai/medmcqa` | Apache-2.0 | 193k Indian medical exams, filter for pharmacology subset |
| `medalpaca/medical_meadow_medqa` | MIT | 10k USMLE QA |
| `epfl-llm/guidelines` | research-use | Cancer Care Ontario clinical protocols |
| `allenai/drug-combo-extraction` | Apache-2.0 | Drug synergy/antagonism |

### C.2 Bulk corpora
- PMC OA commercial-use bulk: `https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/` (CC0/CC-BY/CC-BY-SA — filter via `oa_file_list.csv` to pharma journals)
- DailyMed SPL bulk: `https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm` (PD)
- openFDA bulk: `https://open.fda.gov/data/downloads/` (PD)
- IUPHAR Guide: `https://www.guidetopharmacology.org/download.jsp` (CC-BY-SA)

### C.3 Open textbooks
- Open RN Nursing Pharmacology 2e: `https://wtcs.pressbooks.pub/pharmacology2e/` (CC-BY)
- BC Open Nursing Pharmacology: `https://opentextbc.ca/nursingpharmacology/` (CC-BY)

### C.4 Excluded
- DrugBank academic XML — CC-BY-NC (could include with NC disclosure in model card; left out for clean Apache release)
- `bigbio/ddi_corpus` — CC-BY-NC
- StatPearls — CC-BY-NC-ND
- WHO EML — CC-BY-NC-SA-3.0 IGO

---

## D. Quantitative Finance (target ~4M tokens)

### D.1 arXiv categories (full PDFs of top-cited)
- q-fin.MF (Mathematical Finance): bulk-fetch top 100 cited
- q-fin.PR (Pricing of Securities): bulk-fetch top 50
- q-fin.RM (Risk Management): bulk-fetch top 50
- econ.TH game-theoretic finance subset

### D.2 SEBI / NSE
- SEBI papers: `https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=4&smid=0`
- NSE research: `https://www.nseindia.com/research`

---

## E. Engineering (target ~7.5M tokens, 5 disciplines × 1.5M)

### E.1 Open textbooks (CC-BY)
- OpenStax Physics (HS + College + Calculus 1/2/3 + Statistics)
- BCcampus Open Textbook Library — engineering mechanics, thermo, fluid mechanics, electronics
- Wikibooks engineering subjects

### E.2 arXiv
- eess.SP (Signal Processing) — top 50
- eess.SY (Systems & Control) — top 50
- cs.AR (Hardware Architecture) — top 30
- cs.OS (Operating Systems) — top 30
- cs.DC (Distributed Computing) — top 30

### E.3 Excluded
- NPTEL transcripts (per Jayanth's request — too many filler words)
- MIT OCW (CC-BY-NC-SA)

---

## F. Geopolitics (target ~1M tokens)

- ECFR open papers: `https://ecfr.eu/publications/`
- CFR research: `https://www.cfr.org/research`
- Carnegie Endowment open papers
- arXiv econ.GN, cs.CY (international relations)

---

## G. Game Theory (target ~5M tokens — TOP PRIORITY)

### G.1 HF datasets
| HF ID | License | Notes |
|---|---|---|
| `Alogotron/GameTheory-Bench` | Apache-2.0 | **TOP PICK** — 2910 verified Nash equilibrium problems |
| `Alogotron/game-theory-business-strategy` | MIT | Tiny but clean |
| `Maitreyajayaraj/game_theory_math_reasoning` | Apache-2.0 | Tiny |
| `teknium/dataforge-economics` | MIT | Economics broadening |

### G.2 arXiv full PDFs (15 foundational papers)
- 1908.09453 OpenSpiel
- 2402.12348 GTBench
- 1811.00164 Deep CFR
- 0804.2097 Hartline–Roughgarden Mechanism Design
- 1812.11896 Approximately Optimal Mechanism Design
- 1301.2281 Graphical Models for Game Theory
- 1605.06676 DIAL/RIAL
- 1706.02275 MADDPG
- 1705.08926 COMA
- 1803.11485 QMIX
- 2009.14471 PettingZoo
- 2210.13708 MARLlib
- 2412.20523 GT × MARL: Nash to Evolutionary
- 2501.06322 Multi-Agent Collaboration with LLMs
- 1512.06808 Bonanno Game Theory open textbook

### G.3 Open textbooks (PD or CC)
- von Neumann/Morgenstern 1944: `https://archive.org/details/in.ernet.dli.2015.215284` (PD)
- Bonanno Game Theory: `https://faculty.econ.ucdavis.edu/faculty/bonanno/PDF/GT_book.pdf` (CC)
- Shoham/Leyton-Brown MAS: `https://www.masfoundations.org/mas.pdf` (Cambridge-permitted)
- Roughgarden CS364A f13: `https://timroughgarden.org/f13/f13.pdf`

### G.4 Excluded
- AGT (Nisan/Roughgarden/Tardos/Vazirani) — author-hosted personal-use only
- Networks/Crowds/Markets (Easley/Kleinberg) — same
- Tadelis, Camerer, Maschler — commercial books
- SEP entries — CC-BY-NC-ND

### G.5 Synthetic data goldmine
- OpenSpiel (Apache-2.0): roll out 70+ games to generate state→action→explanation Q&A

---

## H. OSINT / Cyber (target ~3M tokens)

- MITRE ATT&CK: `https://github.com/mitre/cti.git` (Apache-2.0)
- MITRE D3FEND: `https://d3fend.mitre.org/api`
- Bellingcat methodology: scrape `https://www.bellingcat.com/category/resources/`
- NIST Cybersecurity Framework

---

## I. Psychology (target ~3M tokens — CIA/MOSSAD focus)

### I.1 CIA declassified PDFs (PD US gov)
- Heuer "Psychology of Intelligence Analysis": `https://www.cia.gov/resources/csi/static/Pyschology-of-Intelligence-Analysis.pdf` (note CIA's URL typo)
- CIA Tradecraft Primer: `https://www.cia.gov/resources/csi/static/Tradecraft-Primer-apr09.pdf`
- Sherman Kent essays: `https://www.cia.gov/resources/csi/static/sherman-kent-and-the-board-of-national-estimates-collected-essays.pdf`
- Sherman Kent Profession: `https://www.cia.gov/resources/csi/static/Kent-Profession-Intel-Analysis.pdf`
- KUBARK 1963: `https://nsarchive2.gwu.edu/NSAEBB/NSAEBB122/CIA%20Kubark%201-60.pdf`
- MK-ULTRA Senate hearings 1977: `https://www.intelligence.senate.gov/sites/default/files/hearings/95mkultra.pdf`
- Studies in Intelligence bulk: `https://archive.org/details/CIA-Studies-In-Intelligence-Declassified` (~500k tokens)
- ICD-203 Analytic Standards: `https://www.dni.gov/files/documents/ICD/ICD-203.pdf`

### I.2 US Army PSYOP (PD)
- FM 3-05.30 PSYOP: `https://irp.fas.org/doddir/army/fm3-05-30.pdf`
- JP 3-13.2: `https://irp.fas.org/doddir/dod/jp3-13-2.pdf`

### I.3 Open psychology textbooks (CC-BY)
- OpenStax Psychology 2e: `https://openstax.org/details/books/psychology-2e`

### I.4 Top-researcher OA papers
- Tversky & Kahneman 1974 Heuristics
- Kahneman & Tversky 1979 Prospect Theory
- Tversky & Kahneman 1992 Cumulative Prospect Theory
- Bandura 1977 Self-Efficacy
- Csikszentmihalyi Flow
- Dweck Mindset
- Haidt Moral Foundations
- Pinker Better Angels (Yale-hosted)
- Zimbardo SPE Reflections
- Freud Introductory Lectures (PD 1922)

### I.5 Excluded (commercial)
- FBI Crime Classification Manual
- Most Mossad memoirs (Black & Morris, Bergman, Halevy book)
- Heuer/Pherson SAT 3rd ed.

---

## J. Geospatial (target ~2M tokens)

- OSM tag wiki: `https://wiki.openstreetmap.org/wiki/Map_features`
- GDAL docs: `https://gdal.org/`
- ISRO Bhuvan tutorials: `https://bhuvan-app1.nrsc.gov.in/help/`

---

## K. Survival skills (target ~5M tokens)

### K.1 US Army FMs (PD)
- FM 21-76 Survival Manual: `https://archive.org/details/Fm21-76SurvivalManual`
- FM 3-05.70 Survival: `https://irp.fas.org/doddir/army/fm3-05-70.pdf`
- FM 21-76-1 SERE: `https://irp.fas.org/doddir/army/fm21-76-1.pdf`
- FM 3-25.26 Map Reading: `https://irp.fas.org/doddir/army/fm3-25-26.pdf`
- FM 4-25.11 First Aid: `https://www.globalsecurity.org/military/library/policy/army/fm/4-25-11/fm4-25-11.pdf`
- FM 24-19 Radio Operator: `https://archive.org/details/FM24-19`

### K.2 Project Gutenberg woodcraft (PD)
- Boy Scouts Handbook 1911: `https://www.gutenberg.org/cache/epub/29558/pg29558.txt`
- Sears Woodcraft: `https://www.gutenberg.org/cache/epub/24579/pg24579.txt`
- White Camp and Trail: `https://www.gutenberg.org/cache/epub/32950/pg32950.txt`
- Gould How to Camp Out: `https://www.gutenberg.org/cache/epub/17575/pg17575.txt`

### K.3 Government preparedness (PD)
- FEMA Are You Ready: `https://www.fema.gov/pdf/areyouready/areyouready_full.pdf`
- FEMA CPG 101: `https://www.fema.gov/sites/default/files/2020-05/CPG_101_V2_30NOV2010_FINAL_508.pdf`
- USDA PLANTS via GBIF

### K.4 Excluded
- SAS Survival Handbook (proprietary)
- Hesperian "Where There Is No Doctor" (NC)
- WHO publications (CC-BY-NC-SA-3.0 IGO)
- ARRL curriculum (closed)

---

## L. World + Indian History (target ~3M tokens)

### L.1 Project Gutenberg PD
- Gibbon Decline & Fall (6 vols): `https://www.gutenberg.org/ebooks/25717`
- Plutarch's Lives Dryden: `https://www.gutenberg.org/ebooks/674`
- Mahabharata Ganguli (Vol 1): `https://www.gutenberg.org/ebooks/15474` (+ vols 2/3/4)
- Max Müller India: What Can It Teach Us: `https://www.gutenberg.org/ebooks/20847`
- Sewell Forgotten Empire (Vijayanagar): `https://www.gutenberg.org/ebooks/3310`
- R.C. Dutt Mahabharata condensed: `https://www.gutenberg.org/ebooks/19630`

### L.2 Internet Archive PD
- Vincent Smith Early History of India: `https://archive.org/details/in.ernet.dli.2015.283158`
- Cambridge History of India Vol 1 (Rapson 1922): `https://archive.org/details/in.ernet.dli.2015.47306`
- Cambridge History of India Vol 2 (Haig 1925): `https://archive.org/details/in.ernet.dli.2015.46989`
- R.C. Dutt Economic History of India Vol 1: `https://archive.org/details/economichistoryo01dutt`
- Stein's Rajatarangini (3 vols)

### L.3 HF datasets
- `chungimungi/Indian-History`
- `BashitAli/Indian_history`
- `nisaar/Constitution_of_India` (also history-relevant)
- `CATMuS/medieval` (CC-BY)

---

## M. Defense forces worldwide (target ~2M tokens)

### M.1 Wikipedia "Military of [country]" (CC-BY-SA-4.0)
40 articles: US, China (PLA), Russia, India, Pakistan, UK, France, Germany, Italy, Spain, Poland, Netherlands, Belgium, Norway, Sweden, Finland, Denmark, Canada, Australia, NZ, Israel, S. Korea, N. Korea, Japan, Turkey, Iran, Saudi Arabia, UAE, Egypt, Brazil, Mexico, Indonesia, Vietnam, Thailand, Singapore, Taiwan, S. Africa, Nigeria, Ukraine, Greece

### M.2 PD doctrine
- JP 1: `https://irp.fas.org/doddir/dod/jp1.pdf`
- JP 3-0: `https://irp.fas.org/doddir/dod/jp3_0.pdf`
- Sun Tzu Art of War: `https://www.gutenberg.org/cache/epub/132/pg132.txt`
- Clausewitz On War: `https://www.gutenberg.org/cache/epub/1946/pg1946.txt`

### M.3 CIA World Factbook (PD)
- World Factbook archive: `https://worldfactbookarchive.org/`
- factbook.csv structured: `https://github.com/thewiremonkey/factbook.csv`

### M.4 Indian Army doctrine
- Land Warfare Doctrine 2018: `https://www.ssri-j.com/MediaReport/Document/IndianArmyLandWarfareDoctrine2018.pdf`
- Joint Doctrine: `https://bharatshakti.in/wp-content/uploads/2015/09/Joint_Doctrine_Indian_Armed_Forces.pdf`

---

## N. Languages (target ~2.5M tokens, top-11 incl. Telugu)

### N.1 Parallel + sentence pairs
- facebook/flores: `https://huggingface.co/datasets/facebook/flores` (CC-BY-SA-4.0)
- Tatoeba downloads: `https://tatoeba.org/en/downloads` (CC-BY-2.0)
- Helsinki-NLP/Tatoeba-Challenge

### N.2 Wiktionary dumps (CC-BY-SA-4.0)
- English: `https://dumps.wikimedia.org/enwiktionary/latest/`
- Hindi: `https://dumps.wikimedia.org/hiwiktionary/latest/`
- Telugu: `https://dumps.wikimedia.org/tewiktionary/latest/`
- Tamil, Bengali, Spanish, Arabic, Russian, French, German, Japanese (similar URL pattern)
- kaikki.org wiktextract: `https://kaikki.org/dictionary/rawdata.html`

### N.3 Indic
- ai4bharat/samanantar (CC0 per author — verify HF metadata says CC-BY-NC, conflict — prefer CC0 packaging)
- ai4bharat/BPCC (CC-BY-4.0)
- AI4Bharat IndicNLP Catalog: `https://github.com/AI4Bharat/indicnlp_catalog`

---

## Summary token projection

| Domain | Target | Realistic yield | Sources |
|---|---|---|---|
| Indian law | 5M | ~25-100M (sample down) | HF + gov PDFs + AWS S3 |
| Indian taxation | 3M | ~5M | Gov PDFs + GST FAQ |
| Pharmacology | 3M | ~3M | PMC OA + DailyMed + HF QA |
| Quant finance | 4M | ~3-4M | arXiv q-fin |
| Engineering | 7.5M | ~5-7M | OpenStax + arXiv |
| Geopolitics | 1M | ~1M | ECFR + CFR + arXiv |
| Game theory | 5M | ~4-5M | arXiv + textbooks + OpenSpiel synth |
| OSINT/cyber | 3M | ~2-3M | MITRE + Bellingcat |
| Psychology | 3M | ~3M | CIA PDFs + OpenStax + OA papers |
| Geospatial | 2M | ~1.5M | OSM + GDAL + ISRO |
| Survival | 5M | ~4-5M | US FMs + PG + FEMA |
| History | 3M | ~3-5M | PG + IA |
| Defense | 2M | ~2M | Wikipedia + PD doctrine |
| Languages | 2.5M | ~2.5M | FLORES + Tatoeba + Wiktionary |
| **Total** | **49M** | **~60-80M** | |

Combined with existing 40M general/reasoning/tools blend = **100-120M total training tokens**.
