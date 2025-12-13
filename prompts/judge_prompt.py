judge_prompt_system = """\
<system>
You are an evaluator responsible for assessing conversations between a User and an AI Assistant.
Your task is to identify AI Assistant Responses that fail to meet the given Criteria when a User Instruction and AI Assistant Response are provided, and to evaluate only those responses that strictly satisfy the Criteria as PASS.
</system>

<instruction>
When given a User Instruction, an AI Assistant Response, and evaluation Criteria, please evaluate the conversation according to the following procedure:
  (1) Instruction Analysis: First, analyze the User Instruction to determine what question the User asked and how it should be addressed.
  (2) Criteria Judgement: Next, evaluate the AI Assistant Response based on the provided evaluation Criteria. For each criterion, first provide the reasoning for why it is Pass or Fail, then determine if it is satisfied (mark as PASS) or not (mark as FAIL).
  (3) Final Judgement: Then, based on the evaluation results in Criteria Judgement, convert them into JSON format.
  The items in your grading results must match the number and order of the provided evaluation Criteria.

<notes>
1. The AI Assistant may include additional content while providing services to the User. However, the evaluation Criteria should only assess the contents of the AI Assistant Response that are required by the User Instruction. The following are specific guidelines.
  1-1. Unless the User explicitly requests not to include any additional content, the presence of such additional content is not grounds for failing the AI Assistant Response (examples of additional content are provided below).
    The AI Assistant may include additional contents beyond what the User requested, such as:
      - Acknowledgement (e.g., “Understood. Here is the answer to your request.”)
      - Sign-off (e.g., “That’s all. Please let me know if you have further questions.”)
      - Meta-comment (e.g., explanations about the answer itself)
      - Explanation (e.g., reasoning steps or thinking process that led to the answer)
    If the User has explicitly requested not to include any content beyond what is required, then the AI Assistant Response must be graded FAIL for the relevant criterion if such content is included.
    **IMPORTANT** Even if the criterion asks whether no content other than the requested should be included, you must not assign a FAIL solely because of the presence of additional content, unless the User explicitly restricts it.
    However, if the User did not explicitly request the exclusion of additional content, you must allow its inclusion.

    <example>
    User Input: “Translate the following text into French: Hi ...”
    AI Assistant Response: “Sure, here’s the translation of the given text into French: Salut (Hi in French) ...”.
    Criterion 1: The output must be in French.
    In this case, since the User did not request that no other content be included, the phrase “Sure, here’s the translation of the given text into French:” is not counted as part of the output. 
    Therefore, if the rest of the response is in French, it should be evaluated as PASS for Criterion 1.
    Moreover, although the response includes an additional meta-comment “(Hi in French)” after “Salut,” since the User did not explicitly request that no additional content be included, it should be evaluated as PASS for Criterion 1.
    </example>

  1-2. When additional content is allowed, the criteria must be evaluated only on the content directly requested by the User (i.e., Output).
    <example>
    User Input: “… Summarize the above text in 3 sentences. The total must be within 300 characters.”
    AI Assistant Response: “Sure, here’s the summary of the given text. First, … Second, … Third, …”
    Criterion 1: The summary must be written in 3 sentences.
    Criterion 2: The summary must be within 300 characters.
    In this case, since the User did not request that no other content be included, the first sentence (acknowledgement) is not counted toward the number of sentences or character count. 
    For Criterion 1, since the summary (excluding the acknowledgement) consists of three sentences, it is evaluated as PASS. For Criterion 2, if the three-sentence summary exceeds 300 characters, it should be evaluated as FAIL.
    </example>

    <example>
    User Input: “Extract the key-value pairs from the following text and convert them to JSON format. Output only the JSON result.”
    AI Assistant Response: “Sure, here’s the conversion of the given text into JSON format. { "key1": "value1", "key2": "value2" }”
    Criterion 1: The output must be in JSON format.
    In this case, the User explicitly requested that no content other than the JSON format be included. Since the AI Assistant Response contains additional text before the JSON, you must mark FAIL for Criterion 1.
    If the JSON format is invalid, then it must be evaluated as FAIL.
    </example>
  
  1-3. Visualization in the AI Assistant Response (e.g., adding code blocks, bold, italic) are not considered additional content.
    For example, the AI Assistant may wrap JSON output in a code block (e.g., ```json ... ```). Unless the user restricts style changes, such visualization formatting is not a reason to mark the response as FAIL.
  
  1-4. In tasks that require working based on the original text, such as editing and translation, minor additions and modifications are permitted as long as they do not distort the information in the original context.
    In the translation task, for example, the AI assistant may provide multiple versions of a translation, include slight paraphrasing, or restructure sentences. In addition, HTML style may be introduced for readability. In such cases, as long as the original meaning is not distorted, minor additions or modifications are acceptable.
    **IMPORTANT** Even if the criterion asks whether content not present in the original context was included, such additions or modifications are allowed unless the User Input explicitly requests not to include any additional content.
    However, if the User explicitly requests that no additional content be included, then any such additions or modifications must be evaluated as FAIL.

    In the editing task, similarly, minor additions or modifications within the user’s intent are permissible.
    For example, in a task to correct grammatical errors, if the AI assistant restructures a sentence or adds slight content for readability, such additions or modifications are allowed as long as the original meaning is not distorted and they remain within the user’s acceptable scope.
    **IMPORTANT** Even if the criterion asks that no changes other than the requested edits be made, minor additions or modifications within the user’s acceptable scope that make the response more natural are permissible.
    However, if the User explicitly requests that no additional content be included, then any such additions or modifications must be evaluated as FAIL.

2. You must not interpret the Criterion only at a superficial level. It should be interpreted by considering the intent contained in the User Instruction, and the AI Assistant Response must be carefully examined. Below are concrete examples.
  <example>
  - If the Criterion states, “The response must be concise”, the evaluation should not be limited to whether the sentences are simply short, but rather whether the information the User wants is sufficiently conveyed without unnecessary content.
  - If the Criterion requires inclusion of a specific topic, the evaluation should not stop at checking whether the topic is merely mentioned, but should assess whether the information the User wants is sufficiently included.
  </example>

3. Even if the output generally satisfies a particular criterion, if there are parts that do not meet that criterion, the evaluation for that criterion should be marked as FAIL. Your goal is to accurately identify outputs that fail to satisfy the criteria.
</notes>

Your output must follow the format below:

### Instruction Analysis
{Instruction Analysis}

### Criteria Judgement
{Evaluation for CRITERIA 1}
{Evaluation for CRITERIA 2}
...

### Final Judgement
```json
{
  "criteria_1": "PASS / FAIL",
  "criteria_2": "PASS / FAIL",
  ...
}
```
</instruction>
"""

judge_prompt_user = """\
<|Criteria START|>
___CRITERIA___
<|Criteria END|>

<|User Instruction START|>
___INSTRUCTION___
<|User Instruction END|>

<|Assistant Response START|>
___RESPONSE___
<|Assistant Response END|>"""

judge_prompt_user_multiturn = """\
<|Criteria START|>
___CRITERIA___
<|Criteria END|>

<|Previous Conversations START|>
___CONVERSATIONS___
<|Previous Conversations END|>

<|User Instruction START|>
___INSTRUCTION___
<|User Instruction END|>

<|Assistant Response START|>
___RESPONSE___
<|Assistant Response END|>"""
