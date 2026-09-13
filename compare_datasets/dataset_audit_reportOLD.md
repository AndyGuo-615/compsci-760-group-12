# Dataset Audit Report

- **Generated:** 2026-09-12T13:28:09
- **Dataset:** `/var/home/peter/760_Group_Project/DataSet 3 - Sokoto Coventry Fingerprint Dataset/SOCOFing`
- **Mode:** single-dataset audit (Proposal §7.1 / §7.4 / §7.5 + grouping)
- **SSIM backend:** scikit-image (structural_similarity)
- **pHash threshold:** Hamming <= 10
- **dHash threshold:** Hamming <= 12
- **SSIM threshold:** 0.95
- **Runtime:** 390.5 s

---

## 1. Manifest (Proposal §7.1)

- Total images: **55270**
- Readable: **55270**; unreadable: **0**

| Class | Images |
|---|---|
| Altered | 49270 |
| Real | 6000 |

Formats: bmp (55270)

## 2. Exact Duplicates (Proposal §7.4)

- Exact duplicate groups: **11**
- Images in exact duplicate groups: **26**
- Exact groups spanning >1 class: **0**

### Exact duplicate groups (first 11 of 11)

| Group | Size | Classes | Conflict | Files (first 5) |
|---|---|---|---|---|
| EXACT_00001 | 2 | 1 | no | Real/297__M_Left_little_finger.BMP, Real/590__M_Left_little_finger.BMP |
| EXACT_00002 | 2 | 1 | no | Real/511__M_Left_middle_finger.BMP, Real/590__M_Right_little_finger.BMP |
| EXACT_00003 | 2 | 1 | no | Real/511__M_Left_ring_finger.BMP, Real/590__M_Right_middle_finger.BMP |
| EXACT_00004 | 2 | 1 | no | Real/511__M_Left_little_finger.BMP, Real/590__M_Right_ring_finger.BMP |
| EXACT_00005 | 2 | 1 | no | Real/511__M_Right_little_finger.BMP, Real/577__M_Left_little_finger.BMP |
| EXACT_00006 | 2 | 1 | no | Real/553__M_Left_thumb_finger.BMP, Real/554__M_Left_little_finger.BMP |
| EXACT_00007 | 2 | 1 | no | Real/511__M_Left_index_finger.BMP, Real/553__M_Left_little_finger.BMP |
| EXACT_00008 | 2 | 1 | no | Real/363__M_Left_little_finger.BMP, Real/376__F_Left_little_finger.BMP |
| EXACT_00009 | 2 | 1 | no | Real/560__F_Left_little_finger.BMP, Real/585__M_Left_little_finger.BMP |
| EXACT_00010 | 5 | 1 | no | Real/203__M_Left_little_finger.BMP, Real/219__M_Left_little_finger.BMP, Real/249__M_Left_little_finger.BMP, Real/260__M_Left_little_finger.BMP, Real/64__M_Left_little_finger.BMP |
| EXACT_00011 | 3 | 1 | no | Real/204__F_Left_little_finger.BMP, Real/418__F_Left_little_finger.BMP, Real/92__F_Left_little_finger.BMP |

## 3. Near Duplicates (Proposal §7.5)

- Hash-level candidate pairs: **884896**
- Accepted pairs (SSIM >= 0.95): **20245**
- Accepted pairs spanning >1 class: **9910**

### Candidate pairs, highest SSIM first (first 20 of 884896)

| Class A | Class B | File A | File B | pHash | dHash | SSIM | Related |
|---|---|---|---|---|---|---|---|
| Altered | Real | Altered/Altered-Easy/185__M_Left_index_finger_Obl.BMP | Real/185__M_Left_index_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/127__F_Left_thumb_finger_Obl.BMP | Real/127__F_Left_thumb_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/131__M_Right_index_finger_Obl.BMP | Real/131__M_Right_index_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/26__M_Right_little_finger_Obl.BMP | Real/26__M_Right_little_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/212__M_Right_thumb_finger_Obl.BMP | Real/212__M_Right_thumb_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/219__M_Right_ring_finger_Obl.BMP | Real/219__M_Right_ring_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/226__M_Right_little_finger_Obl.BMP | Real/226__M_Right_little_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Real | Altered/Altered-Easy/348__F_Right_thumb_finger_Obl.BMP | Real/348__F_Right_thumb_finger.BMP | 0 | 0 | 1.0000 | yes |
| Altered | Altered | Altered/Altered-Hard/363__M_Left_little_finger_Obl.BMP | Altered/Altered-Medium/363__M_Left_little_finger_Obl.BMP | 2 | 1 | 0.9991 | yes |
| Altered | Real | Altered/Altered-Easy/590__M_Left_middle_finger_Zcut.BMP | Real/590__M_Left_middle_finger.BMP | 0 | 0 | 0.9991 | yes |
| Altered | Altered | Altered/Altered-Hard/226__M_Right_ring_finger_Obl.BMP | Altered/Altered-Medium/226__M_Right_ring_finger_Obl.BMP | 0 | 0 | 0.9991 | yes |
| Altered | Real | Altered/Altered-Easy/585__M_Left_little_finger_Zcut.BMP | Real/560__F_Left_little_finger.BMP | 2 | 0 | 0.9990 | yes |
| Altered | Real | Altered/Altered-Easy/585__M_Left_little_finger_Zcut.BMP | Real/585__M_Left_little_finger.BMP | 2 | 0 | 0.9990 | yes |
| Altered | Real | Altered/Altered-Easy/270__M_Left_index_finger_Zcut.BMP | Real/270__M_Left_index_finger.BMP | 0 | 0 | 0.9989 | yes |
| Altered | Real | Altered/Altered-Easy/354__M_Left_little_finger_Zcut.BMP | Real/354__M_Left_little_finger.BMP | 0 | 0 | 0.9989 | yes |
| Altered | Real | Altered/Altered-Easy/226__M_Left_little_finger_Zcut.BMP | Real/226__M_Left_little_finger.BMP | 0 | 0 | 0.9989 | yes |
| Altered | Altered | Altered/Altered-Hard/270__M_Left_little_finger_Obl.BMP | Altered/Altered-Medium/270__M_Left_little_finger_Obl.BMP | 2 | 0 | 0.9989 | yes |
| Altered | Altered | Altered/Altered-Easy/226__M_Right_little_finger_Obl.BMP | Altered/Altered-Medium/226__M_Right_little_finger_Obl.BMP | 0 | 0 | 0.9988 | yes |
| Altered | Real | Altered/Altered-Medium/226__M_Right_little_finger_Obl.BMP | Real/226__M_Right_little_finger.BMP | 0 | 0 | 0.9988 | yes |
| Altered | Real | Altered/Altered-Easy/590__M_Left_little_finger_Zcut.BMP | Real/297__M_Left_little_finger.BMP | 2 | 2 | 0.9988 | yes |

## 4. Groups (union-find)

- Total groups: **38597**
- Non-singleton groups: **11654**
- Groups spanning >1 class: **5973**

### Non-singleton groups (first 20)

| Group | Size | Classes | Reason | Conflict |
|---|---|---|---|---|
| G00001 | 4 | 2 | near_duplicate | yes |
| G00003 | 3 | 1 | near_duplicate | no |
| G00004 | 2 | 2 | near_duplicate | yes |
| G00007 | 2 | 2 | near_duplicate | yes |
| G00009 | 3 | 2 | near_duplicate | yes |
| G00010 | 4 | 2 | near_duplicate | yes |
| G00013 | 2 | 2 | near_duplicate | yes |
| G00014 | 3 | 2 | near_duplicate | yes |
| G00016 | 3 | 2 | near_duplicate | yes |
| G00019 | 3 | 1 | near_duplicate | no |
| G00020 | 2 | 2 | near_duplicate | yes |
| G00021 | 4 | 2 | near_duplicate | yes |
| G00024 | 2 | 2 | near_duplicate | yes |
| G00025 | 3 | 2 | near_duplicate | yes |
| G00029 | 2 | 2 | near_duplicate | yes |
| G00032 | 2 | 2 | near_duplicate | yes |
| G00035 | 2 | 2 | near_duplicate | yes |
| G00036 | 3 | 2 | near_duplicate | yes |
| G00039 | 3 | 1 | near_duplicate | no |
| G00040 | 2 | 2 | near_duplicate | yes |

## 5. Conclusion

Found 11 exact duplicate group(s) and 20245 accepted near-duplicate pair(s). **5973 group(s) span more than one class** — check for label leakage.

