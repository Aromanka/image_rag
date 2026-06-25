# ECE450 prompt template for backbone model

# Prompt Catalog

文档罗列了项目使用的所有prompt模板和demo case，按照“数据集\-prompt变体”的层级排列。目前prompt包含以下数据集：

- InspecSafe\-V1: binary construction\-safety classification, `safe` / `unsafe`

- ConstructionSite\-10K: rule\-based construction\-site inspection, JSON output

- LabSafety\-v1 generated: binary laboratory hazard classification, `hazardous` / `non-hazardous`

## Shared Variables

|Variable|Meaning|
|---|---|
|`{query}`|User question or task instruction passed by evaluator/API\.|
|`{query_image}`|Image being evaluated\.|
|`{reference_i_image}`|Retrieved image used as a RAG reference\.|
|`{reference_i_caption}`|Retrieved caption, annotation, question, or description\.|
|`{reference_i_label}`|Retrieved ground\-truth label\.|
|`{reference_i_distance}`|Optional vector search distance\.|
|`{top_k}`|Number of retrieved references\. Default is `5`\.|



## InspecSafe\-V1

Problem type: binary visual safety classification\.

Task type in code: `safety judgement`

Default query:

```Plain Text
Is the following image a safe scenario?
```

Output labels:

```Plain Text
safe
unsafe
```



### `inspecsafe.baseline.v1`

Template:

```Plain Text
You are a construction safety visual inspection assistant.

Question for the query image:
{query}

Use only the query image as evidence.

Return your answer in this format:
Query image observations:
Reasoning:
Final label: safe or unsafe
```



### `inspecsafe.rag.v1`

Template:

```Plain Text
System:
You are a construction safety visual inspection assistant. Use the reference images to inform your judgement of the query image.

User content:
[image: {reference_1_image}]
Reference 1: {reference_1_caption} (label: {reference_1_label})

...

[image: {reference_k_image}]
Reference {k}: {reference_k_caption} (label: {reference_k_label})

[image: {query_image}]
Query Image: {query}
Classify ONLY this query image based on the reference examples above.

Return your answer in this format:
Query image observations:
Retrieved evidence:
Reasoning:
Final label: safe or unsafe
```

### `inspecsafe.text_rag.v1`

Template:


```Plain Text
You are a construction safety assistant.

User query:
{query}

Retrieved similar historical cases:
Example 1:
Image path: {reference_1_image_path}
Caption: {reference_1_caption}
Historical safety label: {reference_1_label}

...

Task:
1. Summarize the retrieved evidence.
2. Decide whether the situation is likely safe or unsafe.
3. Explain the reasoning using the retrieved examples.
4. Return a final label: safe or unsafe.
```


## ConstructionSite\-10K


Problem type: construction\-site rule inspection\.


Default query:



```Plain Text
Inspect this construction site image.
```


Output contract:



```JSON
{
  "annotation": "<detailed scene description>",
  "violations": [
    {
      "rule": 1,
      "reason": "<specific visual evidence>"
    }
  ]
}
```

If no violations exist, `violations` must be an empty list\.



### `constructionsite10k.baseline.v1`

Template:

```Plain Text
You are a professional construction site safety inspector with expertise in hazard identification and regulatory compliance.

Carefully analyze the provided construction site image and assess safety compliance step by step.

## Safety Rules

Rule 1 - Personal Protective Equipment (PPE):
All workers on foot must wear: hard hats, clothes covering shoulders and legs, toe-covering shoes. When cutting/welding/grinding/drilling: face shields or safety glasses. At night: high-visibility retroreflective vests.

Rule 2 - Working at Height:
Workers at heights >= 3 meters with unprotected edges must wear a safety harness.

Rule 3 - Edge Protection:
Underground excavations >= 3 meters deep with steep retaining walls require guardrails or warning fences when workers are present.

Rule 4 - Excavator Proximity:
No worker shall appear in the blind spots or within the operation radius of an active excavator, or any excavator with an operator inside.

## Instructions

Step 1 - Scene Description: Describe what you observe including workers, positions, activities, equipment, and environment.
Step 2 - Rule Analysis: For each rule, state whether it is complied with or violated with specific visual evidence.
Step 3 - Output the following JSON only, no extra text:

{
  "annotation": "<detailed scene description>",
  "violations": [
    {
      "rule": <rule_id as integer>,
      "reason": "<specific visual evidence>"
    }
  ]
}

If no violations are found, return an empty list for violations.

Question for the query image:
{query}
```

### `constructionsite10k.rag.v1`


Template:


```Plain Text
System:
{CONSTRUCTIONSITE10K_SYSTEM_PROMPT}

User content:
[image: {reference_1_image}]
Reference 1: {reference_1_annotation}
Ground-truth violation rules: {reference_1_violation_rules}

...

[image: {reference_k_image}]
Reference {k}: {reference_k_annotation}
Ground-truth violation rules: {reference_k_violation_rules}

[image: {query_image}]
Query image task: {query}
Use the reference examples for visual context only. Classify the query image under rules 1-4 and return JSON only.
```



## LabSafety\-v1 Generated



Problem type: synthetic laboratory\-scene hazard classification\.



Default query:



```Plain Text
Classify this laboratory scene as hazardous or non-hazardous.
```



Output labels:



```Plain Text
hazardous
non-hazardous
```



### `lab_safety_gen.baseline.v1`


Template:



```Plain Text
You are a laboratory safety expert reviewing synthetic laboratory-scene images.

Carefully inspect the query image for visible hazards, unsafe handling, missing PPE, unsafe storage, spills, fire or chemical risks, and other laboratory safety issues.

Use the retrieved examples only as reference cases. Classify the query image itself as exactly one of:
- hazardous
- non-hazardous

Return your answer in this format:
Query image observations:
Retrieved evidence:
Reasoning:
Final label: hazardous or non-hazardous

Question for the query image:
{query}
```


### `lab_safety_gen.rag.v1`


Template:



```Plain Text
System:
{LAB_SAFETY_GEN_SYSTEM_PROMPT}

User content:
[image: {reference_1_image}]
Reference 1:
Ground-truth label: {reference_1_label}
Description: {reference_1_description}
Hazards: {reference_1_hazards}
VLM label check: {reference_1_vlm_label}
Agreement flag: {reference_1_agree}

...

[image: {reference_k_image}]
Reference {k}:
Ground-truth label: {reference_k_label}
Description: {reference_k_description}
Hazards: {reference_k_hazards}
VLM label check: {reference_k_vlm_label}
Agreement flag: {reference_k_agree}

[image: {query_image}]
Query image task: {query}
Use the query image as primary evidence. Use the references only to calibrate what hazardous and non-hazardous lab scenes look like.

Return the requested format and end with exactly one final label: hazardous or non-hazardous.
```


## Real RAG Demo Cases


The following cases are sampled from exported demo folders generated by

`utils/evaluate_rag_details.py`\. Each case records the real query image,

retrieved visual references, prompt excerpt, model response, and ground truth\.



### InspecSafe\-V1 Real Cases


#### Case `61`, correct safe prediction



Query image:


\!\[InspecSafe case 61 query\]\(\.\./demo/inspecsafe\_rag\_details/61/query\_image\.jpg\)



Retrieved images:



\!\[InspecSafe 61 retrieved 1\]\(\.\./demo/inspecsafe\_rag\_details/61/retrieved\_01\.jpg\)

\!\[InspecSafe 61 retrieved 2\]\(\.\./demo/inspecsafe\_rag\_details/61/retrieved\_02\.jpg\)

\!\[InspecSafe 61 retrieved 3\]\(\.\./demo/inspecsafe\_rag\_details/61/retrieved\_03\.jpg\)



Real prompt excerpt from `prompt.txt`:



```Plain Text
System:
You are a construction safety visual inspection assistant. Use the reference images to inform your judgement of the query image.

Reference 1: The image shows a conveyor belt transporting what appears to be aggregate material at a construction site. There are at least three workers visible; two are wearing red hard hats and one is wearing a white hard hat. The workers with red hard hats are standing next to machinery that seems to be processing the material, while the worker with the white hard hat is observing or supervising. (label: safe)

Reference 2: The image shows three workers wearing red hard hats standing on a metal platform. (label: safe)

Reference 3: The image shows a conveyor belt system at a construction site with six individuals around it. In the foreground, a person wearing a blue jacket and gloves is standing next to the conveyor belt mechanism. Five other individuals, four wearing red hard hats and one in a grey hat, are standing in a line along the conveyor belt, observing or monitoring the process. (label: safe)

Reference 4: Two workers are visible within a scaffolding structure; one in the foreground is wearing a red hard hat and camouflage jacket, bending over, possibly handling materials, while the other, in the background, is wearing a white hard hat and a reflective vest, standing and facing away. The environment is cluttered with construction materials, including pipes and wooden supports, with the floor appearing to be wet or covered in a thin layer of concrete. (label: safe)

Reference 5: The image shows an underground construction site with scaffolding and support structures. There is one worker visible, wearing a red hard hat and standing on the scaffolding. Various construction materials, such as wooden planks and pipes, are scattered throughout the site. The environment appears to be a tunnel or an underground passage in the process of being built or reinforced. (label: safe)

Query Image: Is the following image a safe scenario?
Classify ONLY this query image based on the reference examples above.

Return your answer in this format:
Query image observations:
Retrieved evidence:
Reasoning:
Final label: safe or unsafe
```



Model response from `response.txt`:



```Plain Text
Query image observations: Workers wearing hard hats and safety gear are present.
Retrieved evidence: The image shows a construction site with workers wearing hard hats and safety gear, which indicates a safe working environment.
Reasoning: The presence of workers wearing appropriate safety equipment suggests that the scenario is considered safe.
Final label: safe
```



Ground truth from `gt_output.txt`:



```Plain Text
safe
```



#### Case `1015`, false safe prediction on unsafe sample



Query image:



\!\[InspecSafe case 1015 query\]\(\.\./demo/inspecsafe\_rag\_details/1015/query\_image\.jpg\)



Retrieved images:



\!\[InspecSafe 1015 retrieved 1\]\(\.\./demo/inspecsafe\_rag\_details/1015/retrieved\_01\.jpg\)

\!\[InspecSafe 1015 retrieved 3\]\(\.\./demo/inspecsafe\_rag\_details/1015/retrieved\_03\.jpg\)

\!\[InspecSafe 1015 retrieved 5\]\(\.\./demo/inspecsafe\_rag\_details/1015/retrieved\_05\.jpg\)



Real prompt excerpt from `prompt.txt`:



```Plain Text
Reference 1: The image shows an underground construction site with scaffolding and support structures. There is one worker visible, wearing a red hard hat and standing on the scaffolding. Various construction materials, such as wooden planks and pipes, are scattered throughout the site. The environment appears to be a tunnel or an underground passage in the process of being built or reinforced. (label: safe)

Reference 2: Two workers wearing yellow hard hats are inside a dimly lit area with earthen walls. The worker on the left is facing a metal structure, while the worker on the right is holding a hose or cable and appears to be engaged in a task. (label: safe)

Reference 3: The image shows three workers at a construction site during nighttime operations. Two workers wearing red hard hats are on the left facing a large concrete bucket in the middle, the third is holding a light source on the right. (label: unsafe)

Reference 4: A worker is standing on a pile of large rocks, looking towards a suspended construction skip or container. Visible in the background are construction materials and possibly part of a structure or scaffolding. (label: unsafe)

Reference 5: The image shows a group of four workers inside what appears to be an underground construction area with rocky terrain. Two workers are wearing hard hats, one in red and the other in white, while the other two are without hard hats. They seem to be engaged in a discussion or assessment of the site. In the background, there is construction equipment, including what looks like a concrete mixer and some vertical rebar structures. (label: unsafe)

Query Image: Is the following image a safe scenario?
Classify ONLY this query image based on the reference examples above.
```



Model response excerpt:



```Plain Text
Query image observations: The image shows a construction site with workers wearing hard hats and safety gear. There is a large concrete block or structure in the foreground, and the workers appear to be engaged in construction activities.
...
Reasoning: Based on the retrieved evidence, most of the images show workers in a safe scenario, with proper safety gear and equipment. However, one image shows a worker standing on a pile of large rocks, which could pose a risk if not properly secured. The other images do not provide enough context to determine the safety of the scenario.
Final label: safe
```



Ground truth:



```Plain Text
unsafe
```



Observation for prompt iteration: this real failure suggests that retrieved

mixed\-label evidence can pull the model toward a majority or generic PPE

judgement unless the prompt emphasizes query\-image hazards over retrieved\-label

frequency\.



### ConstructionSite\-10K Real Cases



Source folder: `demo/constructionsite_rag_details`



#### Case `0000001`, correct no\-violation prediction



Query image:



\!\[ConstructionSite case 0000001 query\]\(\.\./demo/constructionsite\_rag\_details/0000001/query\_image\.jpg\)



Retrieved images:



\!\[ConstructionSite 0000001 retrieved 1\]\(\.\./demo/constructionsite\_rag\_details/0000001/retrieved\_01\.jpg\)

\!\[ConstructionSite 0000001 retrieved 2\]\(\.\./demo/constructionsite\_rag\_details/0000001/retrieved\_02\.jpg\)

\!\[ConstructionSite 0000001 retrieved 3\]\(\.\./demo/constructionsite\_rag\_details/0000001/retrieved\_03\.jpg\)



Real prompt excerpt from `prompt.txt`:



```Plain Text
System:
You are a professional construction site safety inspector with expertise in hazard identification and regulatory compliance.

Rule 1 - Personal Protective Equipment (PPE): ...
Rule 2 - Working at Height: ...
Rule 3 - Edge Protection: ...
Rule 4 - Excavator Proximity: ...

Reference 1: The image shows a mobile crane mounted on a truck chassis, positioned in the center and facing towards the right of the frame. The crane is predominantly yellow with a red telescopic boom, which is retracted. The truck has multiple axles and is parked on a dirt surface with visible tire tracks. In the background, there is a construction site with various materials and equipment, including what appears to be steel beams on the ground and another crane in the distance.
Ground-truth violation rules: none

Reference 2: A mobile crane is positioned in the foreground with its boom partially extended towards the right side of the frame. The crane is mounted on a multi-axle truck with a yellow cab and a red and yellow crane body. In the background, a construction site is visible with a dirt terrain and several construction materials scattered around.
Ground-truth violation rules: none

Reference 3: A mobile crane is parked on a dirt surface with its boom retracted and resting on the support structure above the truck cab.
Ground-truth violation rules: none

Query image task: Inspect this construction site image.
Use the reference examples for visual context only. Classify the query image under rules 1-4 and return JSON only.
```



Model response:



```JSON
{
  "annotation": "A mobile crane is parked on a dirt surface with its boom retracted and resting on the support structure above the truck cab. The crane is yellow and red, with six wheels visible on the side closest to the camera. In the background, there is a cleared area with earth mounds and construction activity is suggested by the presence of what appears to be a tower crane in the far distance.",
  "violations": []
}
```



Ground truth:



```JSON
{
  "annotation": "The rear view of a mobile crane.",
  "violations": []
}
```



#### Case `0000005`, missed PPE violation



Query image:



\!\[ConstructionSite case 0000005 query\]\(\.\./demo/constructionsite\_rag\_details/0000005/query\_image\.jpg\)



Retrieved images:



\!\[ConstructionSite 0000005 retrieved 1\]\(\.\./demo/constructionsite\_rag\_details/0000005/retrieved\_01\.jpg\)

\!\[ConstructionSite 0000005 retrieved 2\]\(\.\./demo/constructionsite\_rag\_details/0000005/retrieved\_02\.jpg\)

\!\[ConstructionSite 0000005 retrieved 3\]\(\.\./demo/constructionsite\_rag\_details/0000005/retrieved\_03\.jpg\)



Real prompt excerpt from `prompt.txt`:



```Plain Text
Reference 1: Two excavators are positioned in the center of the image, with one appearing to be in operation as indicated by the raised bucket. The terrain is uneven and consists of dirt, indicative of an excavation site. Trees and buildings can be seen in the distance, suggesting the site is located near a developed area.
Ground-truth violation rules: none

Reference 2: The image shows a blue Kobelco excavator with the number 55 on its side, positioned on a mound of dirt within a construction area. The excavator's arm is extended with the bucket resting on the ground.
Ground-truth violation rules: none

Reference 3: In the center of the image, an excavator with a yellow arm and a muddy bucket is positioned on a dirt surface, with its boom partially extended.
Ground-truth violation rules: none

Query image task: Inspect this construction site image.
Use the reference examples for visual context only. Classify the query image under rules 1-4 and return JSON only.
```



Model response:



```JSON
{
  "annotation": "Two excavators are positioned in the center of the image, with one appearing to be in operation as indicated by the raised bucket. The terrain is uneven and consists of dirt, indicative of an excavation site. Trees and buildings can be seen in the distance, suggesting the site is located near a developed area.",
  "violations": []
}
```



Ground truth:



```JSON
{
  "annotation": "An excavator is in the center of the image. There is a person walking on the left, and a mobile crane on the right. The mobile crane is hidden by the excavator, but the boom and hook block is visible. A river is in the background.",
  "violations": [
    {
      "rule": 1,
      "reason": "Person on the left not using PPE."
    }
  ]
}
```



Observation for prompt iteration: retrieved examples were all no\-violation

excavator scenes, and the model copied the visual pattern from references while

missing the query\-specific worker/PPE evidence\.



### LabSafety\-v1 Generated Real Cases



Source folder: `demo/lab_safety_gen_rag_details`



#### Case `ls_bench_0021__02`, correct hazardous prediction



Query image:



\!\[LabSafety generated case ls\_bench\_0021\_\_02 query\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0021\_\_02/query\_image\.png\)



Retrieved images:



\!\[LabSafety generated 0021 retrieved 1\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0021\_\_02/retrieved\_01\.png\)

\!\[LabSafety generated 0021 retrieved 2\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0021\_\_02/retrieved\_02\.png\)

\!\[LabSafety generated 0021 retrieved 3\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0021\_\_02/retrieved\_03\.png\)



Real prompt excerpt from `prompt.txt`:



```Plain Text
System:
You are a laboratory safety expert reviewing synthetic laboratory-scene images.

Carefully inspect the query image for visible hazards, unsafe handling, missing PPE, unsafe storage, spills, fire or chemical risks, and other laboratory safety issues.

Reference 1:
Ground-truth label: hazardous
Description: In the laboratory scenario, the potential hazards in the image include the researcher not wearing eye protection while handling chemical substances and the inhalation of chemical vapors rising near his face. Therefore, the scene is classified as HAZARDOUS.
Hazards: Inhalation of pesticide vapor due to a faulty fume hood
VLM label check: hazardous
Agreement flag: True

Reference 2:
Ground-truth label: hazardous
Description: In the laboratory scenario, the potential hazards in the image include accidental spill of a harmful chemical solution onto a researcher's lab coat. Therefore, the scene is classified as HAZARDOUS.
Hazards: accidental spill of a harmful chemical solution onto a researcher's lab coat
VLM label check: hazardous
Agreement flag: True

Reference 3:
Ground-truth label: non-hazardous
Description: In the laboratory scenario, no safety violations are visible. Therefore, the scene is classified as SAFE.
VLM label check: non-hazardous
Agreement flag: True

Query image task: Classify this laboratory scene as hazardous or non-hazardous.
Use the query image as primary evidence. Use the references only to calibrate what hazardous and non-hazardous lab scenes look like.
```



Model response:



```Plain Text
Query image observations: The image shows a researcher wearing safety glasses and gloves, working at a lab bench with various equipment and chemicals. There appears to be a spill of a dark substance on the benchtop. The researcher is handling glassware and solutions.

Retrieved evidence:
- Reference 1: The image shows a researcher not wearing eye protection while handling chemical substances and the inhalation of chemical vapors rising near his face.
- Reference 2: The image shows an accidental spill of a harmful chemical solution onto a researcher's lab coat.
- Reference 4: The image shows a chemical spill onto personal protective equipment.
- Reference 5: The image shows direct UV light exposure causing localized skin redness and discomfort due to insufficient protective measures.

Reasoning: The presence of a spill on the benchtop indicates a potential hazard. While the researcher is wearing gloves and safety glasses, the spill itself represents a risk of chemical exposure. Considering the previous examples, a spill is a clear indicator of a hazardous situation.

Final label: hazardous
```



Ground truth:



```Plain Text
hazardous
```



#### Case `ls_bench_0025__01`, correct non\-hazardous prediction



Query image:



\!\[LabSafety generated case ls\_bench\_0025\_\_01 query\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0025\_\_01/query\_image\.png\)



Retrieved images:



\!\[LabSafety generated 0025 retrieved 1\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0025\_\_01/retrieved\_01\.png\)

\!\[LabSafety generated 0025 retrieved 2\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0025\_\_01/retrieved\_02\.png\)

\!\[LabSafety generated 0025 retrieved 3\]\(\.\./demo/lab\_safety\_gen\_rag\_details/ls\_bench\_0025\_\_01/retrieved\_03\.png\)



Real prompt excerpt from `prompt.txt`:



```Plain Text
Reference 1:
Ground-truth label: non-hazardous
Description: In the laboratory scenario, no safety violations are visible. Therefore, the scene is classified as SAFE.
VLM label check: non-hazardous
Agreement flag: True

Reference 2:
Ground-truth label: non-hazardous
Description: In the laboratory scenario, the potential hazards in the image include the researcher not wearing eye protection. Therefore, the scene is classified as HAZARDOUS.
VLM label check: hazardous
Agreement flag: True

Reference 3:
Ground-truth label: non-hazardous
Description: In the laboratory scenario, no safety violations are visible. Therefore, the scene is classified as SAFE.
VLM label check: non-hazardous
Agreement flag: True

Query image task: Classify this laboratory scene as hazardous or non-hazardous.
Use the query image as primary evidence. Use the references only to calibrate what hazardous and non-hazardous lab scenes look like.
```



Model response:



```Plain Text
Query image observations: The image shows a scientist wearing gloves and a lab coat working inside a biosafety cabinet. There are various lab equipment and supplies visible, including centrifuges, incubators, and racks of petri dishes. The lab appears generally organized and clean.

Retrieved evidence: References 1, 2, 3, 4, and 5 all depict laboratory scenes classified as non-hazardous. These images show researchers wearing appropriate PPE (lab coats, gloves, masks), using safety equipment (biosafety cabinets), and maintaining a generally tidy workspace.

Reasoning: Based on the references, this image appears to represent a standard, safe laboratory setting. The scientist is wearing gloves and a lab coat, and the biosafety cabinet is in use, suggesting appropriate safety precautions are being taken. There are no immediately visible hazards like spills, missing PPE, or unsafe storage.

Final label: non-hazardous
```



Ground truth:



```Plain Text
non-hazardous
```



Observation for prompt iteration: this case includes a useful metadata conflict:

Reference 2 has `Ground-truth label: non-hazardous` but a VLM label check and

description that say hazardous\. The current prompt exposes both fields, so future

ablations can test whether hiding `vlm_label` or inconsistent descriptions

improves stability\.



## Candidate Prompt Slots



The following slots are intentionally reserved for future experiments\.



### `inspecsafe.ablation_no_reasoning.v1`



Purpose: force a short answer for binary classification\.



Template:



```Plain Text
Classify the query image as safe or unsafe.
Return exactly one word: safe or unsafe.
```



### `constructionsite10k.ablation_no_reference_labels.v1`



Purpose: test whether RAG references help visually without exposing retrieved

ground\-truth rule labels\.



Template change:



```Plain Text
Reference {i}: {reference_i_annotation}
```



Remove:



```Plain Text
Ground-truth violation rules: {reference_i_violation_rules}
```



### `lab_safety_gen.ablation_label_only.v1`



Purpose: force a label\-only output for robust parsing\.



Template:



```Plain Text
Inspect the query laboratory image.
Return exactly one label and no other text:
hazardous
non-hazardous
```



## Maintenance Notes



- Keep templates synchronized with `rag_answer.py` and `vlm_inference.py`\.

- Do not delete old prompt variants after experiments; mark them as deprecated instead\.

- When adding examples, include sample ID, image path, expected label, and any retrieved reference metadata\.

- For RAG variants, always state whether retrieved labels are exposed to the model\.

- For parser\-sensitive experiments, document the exact output contract\.

