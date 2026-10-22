# Requirements Document

## Introduction

ShelfSentry AI is an agentic retail shelf inspection system that combines classical computer vision with a bounded active-perception agent loop. The system accepts retail shelf images and short videos, runs substantive OpenCV 5 analysis to produce structured visual findings with quantified uncertainty, then drives a Strands Agents loop that causally selects follow-up perception actions until it reaches an evidence-backed final report. A Model Context Protocol (MCP) server exposes a restricted set of vision tools to the agent without granting broad system access.

The system is built greenfield and runs in two explicit modes. Local/Mock Mode runs the full pipeline without AWS credentials using local sample inputs, real OpenCV analysis, and a deterministic or injected agent model, and is the primary mode for tests and the demo. AWS Mode uses deployed infrastructure (S3, Lambda, API Gateway HTTP API, DynamoDB, CloudWatch) provisioned exclusively through CloudFormation, with Amazon Bedrock as the primary agent decision path and a deterministic policy as a thin safe fallback.

The system is held to strict honesty and safety guardrails: it never fabricates metrics or confidence scores, never silently deletes user data, never embeds secrets, and never claims that an empty-looking shelf region proves an out-of-stock condition without documenting visual caveats such as shadow, occlusion, dividers, or image quality.

## Glossary

- **ShelfSentry_System**: The complete ShelfSentry AI application, including vision pipeline, agent loop, MCP server, data layer, frontend, and infrastructure.
- **Vision_Pipeline**: The OpenCV 5 based component that validates, preprocesses, and analyzes images and video frames and produces structured visual findings.
- **Agent_Loop**: The Strands Agents based bounded active-perception component that selects follow-up actions based on visual findings and uncertainty.
- **MCP_Server**: The Model Context Protocol server that exposes restricted vision tools to the Agent_Loop.
- **Data_Layer**: The component responsible for typed schemas, validation, serialization, and persistence of inspection data.
- **Frontend**: The React, TypeScript, Vite, and Tailwind web application through which a User interacts with the ShelfSentry_System.
- **Infrastructure_Stack**: The CloudFormation defined AWS resources for AWS Mode.
- **User**: A person who submits shelf inspections and reviews results through the Frontend.
- **Reviewer**: A person who performs human review of inspections the Agent_Loop flags for human review.
- **Inspection_Request**: A typed input describing an image or video to inspect and its parameters.
- **Vision_Finding**: A typed structured result from the Vision_Pipeline for a single input, containing candidate regions, metrics, and uncertainty.
- **Candidate_Region**: A single region of interest within an image, with a bounding box, measured visual metric, and heuristic score.
- **Agent_Decision**: A typed record of one action the Agent_Loop selected, including rationale.
- **Tool_Execution**: A typed record of one tool call the Agent_Loop made and the result returned.
- **Inspection_Report**: The typed evidence-backed final output of an inspection.
- **Evaluation_Result**: A typed record of measured performance against a labeled evaluation set.
- **Local_Mock_Mode**: The operating mode that runs without AWS credentials using local inputs, real OpenCV, and a deterministic or injected agent model.
- **AWS_Mode**: The operating mode that uses deployed AWS infrastructure and Amazon Bedrock when available.
- **Bedrock_Path**: The primary agent decision path backed by Amazon Bedrock.
- **Fallback_Policy**: The deterministic decision policy used when the Bedrock_Path is unavailable or errors.
- **Empty_Space_Ratio**: A measured visual metric expressing the proportion of a Candidate_Region estimated to be empty shelf space.
- **Heuristic_Score**: A rule-derived score from classical computer vision, kept distinct from a calibrated probability.
- **Uncertainty_Value**: A quantified confidence or uncertainty measure attached to a Vision_Finding or Candidate_Region, with a documented interpretation.
- **Annotated_Output**: An output image with drawn annotations marking candidate regions and metrics.
- **Decision_Trace**: The ordered record of Agent_Decision and Tool_Execution entries for one inspection.
- **Config_Switch**: The explicit configuration setting that selects Local_Mock_Mode or AWS_Mode.

## Requirements

### Requirement 1: Input Validation

**User Story:** As a User, I want uploaded images and videos validated before processing, so that invalid inputs are rejected with clear feedback instead of causing downstream failures.

#### Acceptance Criteria

1. WHEN an Inspection_Request is submitted with an image file, THE Vision_Pipeline SHALL verify that the file is a decodable image before analysis.
2. WHEN an Inspection_Request is submitted with a video file, THE Vision_Pipeline SHALL verify that the file is a decodable video before analysis.
3. IF a submitted file is empty, THEN THE Vision_Pipeline SHALL reject the Inspection_Request and return a validation error identifying the empty-file condition.
4. IF a submitted file cannot be decoded as a valid image or video, THEN THE Vision_Pipeline SHALL reject the Inspection_Request and return a validation error identifying the corrupt-or-invalid condition.
5. IF a submitted file has a content type that is not in the configured set of accepted content types, THEN THE Vision_Pipeline SHALL reject the Inspection_Request and return a validation error identifying the unsupported content type.
6. IF a submitted file exceeds the configured maximum size, THEN THE Vision_Pipeline SHALL reject the Inspection_Request and return a validation error identifying the size limit.
7. WHEN an Inspection_Request passes all validation checks, THE Vision_Pipeline SHALL accept the Inspection_Request for analysis.

### Requirement 2: Image Preprocessing and OpenCV 5 Analysis

**User Story:** As a User, I want the system to run substantive OpenCV 5 analysis on shelf images, so that candidate empty regions are detected using documented computer vision techniques.

#### Acceptance Criteria

1. THE Vision_Pipeline SHALL perform analysis using OpenCV version 5 and SHALL record the exact OpenCV version used in each Vision_Finding.
2. WHEN an accepted image is analyzed, THE Vision_Pipeline SHALL preprocess the image and perform shelf-region analysis, edge analysis, contour analysis, and morphological operations.
3. WHEN shelf-region analysis completes, THE Vision_Pipeline SHALL compute a Heuristic_Score for each Candidate_Region.
4. WHEN region analysis completes, THE Vision_Pipeline SHALL produce an Annotated_Output image marking each Candidate_Region.
5. THE Vision_Pipeline SHALL label all classical computer vision results as estimates within the Vision_Finding.
6. THE Vision_Pipeline SHALL keep the Heuristic_Score distinct from any calibrated probability in the Vision_Finding.

### Requirement 3: Video Frame Sampling

**User Story:** As a User, I want the system to sample suitable frames from short videos, so that video inspections produce findings comparable to image inspections within processing limits.

#### Acceptance Criteria

1. WHEN an accepted video is analyzed, THE Vision_Pipeline SHALL sample frames from the video up to the configured maximum frame count.
2. WHEN frames are sampled, THE Vision_Pipeline SHALL assign a unique frame identifier to each sampled frame.
3. WHEN frames are sampled, THE Vision_Pipeline SHALL analyze each sampled frame using the same OpenCV 5 analysis defined for images.
4. WHERE a video contains more frames than the configured maximum frame count, THE Vision_Pipeline SHALL select frames according to a documented sampling strategy that stays within the configured limit.

### Requirement 4: Structured Visual Findings

**User Story:** As a User, I want vision analysis to produce structured findings with measurements, uncertainty, and evidence references, so that results are interpretable and traceable.

#### Acceptance Criteria

1. WHEN analysis completes for an input, THE Vision_Pipeline SHALL produce a Vision_Finding containing zero or more Candidate_Region entries.
2. THE Vision_Pipeline SHALL include in each Candidate_Region a bounding box, a measured Empty_Space_Ratio or a documented alternative visual metric, a Heuristic_Score, and an Uncertainty_Value.
3. THE Vision_Pipeline SHALL include in each Vision_Finding a documented interpretation for each Uncertainty_Value.
4. WHEN a Candidate_Region derives from a video frame, THE Vision_Pipeline SHALL include the frame identifier in the Candidate_Region.
5. THE Vision_Pipeline SHALL include in each Vision_Finding an evidence reference to the Annotated_Output for each analyzed input.
6. THE Vision_Pipeline SHALL include in each Vision_Finding a suggested follow-up action for the Agent_Loop.
7. THE Vision_Pipeline SHALL record a visual-caveat note for each Candidate_Region identifying that an empty-looking region may result from shadow, occlusion, dividers, or image quality rather than an out-of-stock condition.

### Requirement 5: Reproducible Annotation Coordinates

**User Story:** As a User, I want annotation coordinates to be consistent and reproducible, so that the same input always yields the same annotated regions.

#### Acceptance Criteria

1. WHEN the same input is analyzed with the same configuration more than once, THE Vision_Pipeline SHALL produce identical bounding box coordinates for each Candidate_Region across runs.
2. THE Vision_Pipeline SHALL express every bounding box in a single documented coordinate system relative to the analyzed input dimensions.
3. WHEN an Annotated_Output is produced, THE Vision_Pipeline SHALL place annotations at coordinates that match the bounding boxes recorded in the Vision_Finding.

### Requirement 6: Bounded Active-Perception Agent Loop

**User Story:** As a User, I want an agent to actively select follow-up perception actions based on visual findings, so that the system gathers additional evidence before reaching a conclusion.

#### Acceptance Criteria

1. WHEN the Agent_Loop receives a Vision_Finding with its Uncertainty_Value, THE Agent_Loop SHALL select the next action from the set {inspect_crop, analyze_another_frame, request_better_image, request_human_review, finalize_report}.
2. WHEN two Vision_Finding inputs differ, THE Agent_Loop SHALL be capable of selecting different next actions such that the initial visual findings causally influence the selected action.
3. WHEN the Agent_Loop selects an action, THE Agent_Loop SHALL record an Agent_Decision containing the selected action and its rationale.
4. WHEN the Agent_Loop invokes a tool, THE Agent_Loop SHALL validate the tool arguments before execution and SHALL reject the invocation if the arguments are invalid.
5. WHEN a tool returns a result, THE Agent_Loop SHALL record a Tool_Execution containing the tool invoked, the arguments, and the result, and SHALL feed the returned evidence into the next iteration.
6. THE Agent_Loop SHALL limit iterations to a configurable maximum with a default of 4 iterations.
7. IF the Agent_Loop reaches the configured maximum iteration count, THEN THE Agent_Loop SHALL stop iterating and produce an Inspection_Report.
8. WHEN the Agent_Loop determines that sufficient evidence has been gathered, THE Agent_Loop SHALL stop iterating and produce an Inspection_Report.
9. WHEN the Agent_Loop selects request_human_review, THE Agent_Loop SHALL stop iterating and produce an Inspection_Report marked as requiring human review.
10. WHEN the Agent_Loop produces an Inspection_Report, THE Agent_Loop SHALL reference the Decision_Trace evidence that supports the final recommendation.

### Requirement 7: Agent Decision Backend and Safe Fallback

**User Story:** As a User, I want the agent to use a Bedrock-backed decision path by default and fall back safely when the model is unavailable, so that inspections continue without silent failure.

#### Acceptance Criteria

1. WHILE AWS_Mode is active and the Bedrock_Path is available, THE Agent_Loop SHALL use the Bedrock_Path as the primary decision path for selecting the next action.
2. IF the Bedrock_Path is unavailable or returns an error, THEN THE Agent_Loop SHALL use the Fallback_Policy to select the next action and SHALL record in the Agent_Decision that the Fallback_Policy was used.
3. WHILE Local_Mock_Mode is active, THE Agent_Loop SHALL support an injected or mocked decision model so that causality tests run without AWS credentials.
4. THE Agent_Loop SHALL produce a deterministic action selection from the Fallback_Policy for a given Vision_Finding and Decision_Trace.

### Requirement 8: MCP Server Tool Exposure

**User Story:** As a developer, I want a minimal MCP server that exposes validated vision tools to the agent, so that the agent accesses perception capabilities through a controlled interface.

#### Acceptance Criteria

1. THE MCP_Server SHALL be implemented using a maintained Python MCP SDK and SHALL use a transport compatible with the Agent_Loop.
2. THE MCP_Server SHALL expose the tools analyze_shelf_image, inspect_region_crop, compare_shelf_frames, and get_annotated_evidence.
3. WHEN an MCP tool is invoked, THE MCP_Server SHALL validate the tool arguments against the tool schema before execution and SHALL reject invocations with invalid arguments.
4. THE MCP_Server SHALL reuse the Vision_Pipeline functions for image processing and SHALL NOT implement duplicate image-processing logic.

### Requirement 9: MCP Security Boundary

**User Story:** As a security owner, I want the MCP server to enforce a strict capability boundary, so that exposing vision tools does not grant the agent broad system access.

#### Acceptance Criteria

1. THE MCP_Server SHALL restrict tool capabilities so that no tool exposes unrestricted AWS credentials.
2. THE MCP_Server SHALL restrict tool capabilities so that no tool exposes unrestricted filesystem access.
3. THE MCP_Server SHALL restrict tool capabilities so that no tool fetches arbitrary user-supplied URLs.
4. THE MCP_Server SHALL restrict tool capabilities so that no tool executes arbitrary shell commands.

### Requirement 10: Typed Data Contracts

**User Story:** As a developer, I want typed and validated data contracts across the system, so that components exchange consistent, serializable, versioned data.

#### Acceptance Criteria

1. THE Data_Layer SHALL define typed schemas for Inspection_Request, Vision_Finding, Candidate_Region, Agent_Decision, Tool_Execution, Inspection_Report, and Evaluation_Result.
2. WHEN data is received at a component boundary, THE Data_Layer SHALL validate the data against its schema and SHALL reject data that fails validation with an error identifying the failing field.
3. WHEN a typed object is serialized to JSON, THE Data_Layer SHALL produce consistent JSON serialization for equivalent objects.
4. WHEN the ShelfSentry_System returns an API response, THE Data_Layer SHALL include an API version identifier in the response.
5. IF schema validation fails, THEN THE Data_Layer SHALL return a structured error describing the validation failure.

### Requirement 11: Inspection Page

**User Story:** As a User, I want an inspection page to upload inputs and start an inspection, so that I can submit shelf media and monitor progress.

#### Acceptance Criteria

1. THE Frontend SHALL provide an inspection page that accepts an image or video upload and provides an example mode using bundled sample inputs.
2. WHEN a User selects a file, THE Frontend SHALL validate the file type and size against the configured limits before submission and SHALL display a validation message when the file is rejected.
3. WHEN a User starts an inspection, THE Frontend SHALL display inspection progress until a result or error is available.
4. IF an inspection returns an error, THEN THE Frontend SHALL display an error state describing the failure.
5. WHILE no inspection has been started, THE Frontend SHALL display an empty state on the inspection page.
6. THE Frontend SHALL conform to WCAG 2.1 Level AA success criteria on the inspection page.

### Requirement 12: Results Page

**User Story:** As a User, I want a results page that shows the original and annotated images with the agent's reasoning, so that I can understand the finding and the recommendation.

#### Acceptance Criteria

1. WHEN an Inspection_Report is available, THE Frontend SHALL display the original input and the Annotated_Output side by side with highlighted Candidate_Region entries.
2. THE Frontend SHALL display for each Candidate_Region the measured metric and its Uncertainty_Value with the documented interpretation.
3. THE Frontend SHALL display the agent initial finding, the selected next action with its rationale, and the recorded Tool_Execution results.
4. THE Frontend SHALL display the final recommendation and the human-review status of the Inspection_Report.
5. THE Frontend SHALL conform to WCAG 2.1 Level AA success criteria on the results page.

### Requirement 13: History Page

**User Story:** As a User, I want a history page listing previous inspections, so that I can find and review past results.

#### Acceptance Criteria

1. THE Frontend SHALL provide a history page listing previous inspections with status and timestamp for each inspection.
2. THE Frontend SHALL provide search and filter controls on the history page.
3. WHEN a User selects an inspection from the history list, THE Frontend SHALL display the detail view for that inspection.
4. WHILE no previous inspections exist, THE Frontend SHALL display an empty state on the history page.
5. THE Frontend SHALL conform to WCAG 2.1 Level AA success criteria on the history page.

### Requirement 14: Evaluation Page

**User Story:** As a User, I want an evaluation page showing measured performance, so that I can assess system accuracy without being misled by fabricated numbers.

#### Acceptance Criteria

1. THE Frontend SHALL provide an evaluation page displaying test-set performance derived from Evaluation_Result records.
2. THE Frontend SHALL display false-positive examples, false-negative examples, agent decision correctness, and latency on the evaluation page.
3. THE Frontend SHALL display documented failure cases and limitations on the evaluation page.
4. THE Frontend SHALL display only measured metrics from recorded Evaluation_Result data and SHALL NOT display fabricated metrics.
5. WHEN displaying evaluation records, THE Frontend SHALL distinguish demo records from real records.
6. THE Frontend SHALL conform to WCAG 2.1 Level AA success criteria on the evaluation page.

### Requirement 15: Local/Mock Mode

**User Story:** As a developer, I want a local mock mode that runs without AWS credentials, so that I can run the full pipeline for tests and the demo offline.

#### Acceptance Criteria

1. WHILE Local_Mock_Mode is active, THE ShelfSentry_System SHALL run the full inspection pipeline without AWS credentials.
2. WHILE Local_Mock_Mode is active, THE ShelfSentry_System SHALL use local sample inputs and perform real OpenCV 5 analysis.
3. WHILE Local_Mock_Mode is active, THE ShelfSentry_System SHALL use a deterministic or injected agent model for Agent_Loop decisions.
4. WHILE Local_Mock_Mode is active, THE ShelfSentry_System SHALL store inspection results in local storage.

### Requirement 16: AWS Mode

**User Story:** As an operator, I want an AWS mode that uses deployed infrastructure, so that inspections run against real cloud services with real integrations.

#### Acceptance Criteria

1. WHILE AWS_Mode is active, THE ShelfSentry_System SHALL use the deployed API Gateway HTTP API, S3 storage, and DynamoDB storage for inspection data, results, and Decision_Trace records.
2. WHILE AWS_Mode is active, THE ShelfSentry_System SHALL use the real Strands Agents integration and SHALL use the Bedrock_Path when available.
3. IF an AWS operation fails during an inspection in AWS_Mode, THEN THE ShelfSentry_System SHALL apply the configured failure handling and retry behavior and SHALL surface a failure status when retries are exhausted.
4. WHILE AWS_Mode is active, THE ShelfSentry_System SHALL NOT substitute mock results for real results.

### Requirement 17: Mode Configuration Switch

**User Story:** As an operator, I want an explicit configuration switch between modes, so that the active mode is unambiguous and misconfiguration fails loudly.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL select between Local_Mock_Mode and AWS_Mode using an explicit Config_Switch.
2. IF AWS_Mode is selected and required AWS configuration is missing, THEN THE ShelfSentry_System SHALL fail startup with a message identifying the missing configuration.
3. WHILE AWS_Mode is selected, THE ShelfSentry_System SHALL NOT silently substitute Local_Mock_Mode results.

### Requirement 18: Storage Infrastructure

**User Story:** As an operator, I want secure storage provisioned through CloudFormation, so that inspection media and data are stored privately and encrypted.

#### Acceptance Criteria

1. THE Infrastructure_Stack SHALL define all AWS resources using CloudFormation exclusively.
2. THE Infrastructure_Stack SHALL provision an S3 bucket that is private, encrypted at rest, and configured to block public access.
3. WHEN a large file upload is required, THE ShelfSentry_System SHALL use a presigned URL for the upload to S3.
4. THE Infrastructure_Stack SHALL provision DynamoDB storage for inspection status, results, and Decision_Trace records.
5. THE Infrastructure_Stack SHALL define configurable resource names for provisioned resources.

### Requirement 19: Compute and API Infrastructure

**User Story:** As an operator, I want compute and API infrastructure provisioned through CloudFormation with explicit limits, so that the system runs with bounded resources and observability.

#### Acceptance Criteria

1. THE Infrastructure_Stack SHALL provision Lambda functions and an API Gateway HTTP API for the inspection API.
2. THE Infrastructure_Stack SHALL define explicit timeout and memory settings for each Lambda function.
3. THE Infrastructure_Stack SHALL provision CloudWatch logs with an explicit retention setting and SHALL define basic alarms.
4. THE Infrastructure_Stack SHALL scaffold a thin asynchronous job path for longer videos using a DynamoDB job-status record and a worker entry point.
5. WHEN an asynchronous job is submitted, THE ShelfSentry_System SHALL record the job status in DynamoDB so that status can be polled.

### Requirement 20: Infrastructure Security and Least Privilege

**User Story:** As a security owner, I want least-privilege IAM and no embedded secrets, so that the deployed system limits blast radius and avoids credential leaks.

#### Acceptance Criteria

1. THE Infrastructure_Stack SHALL grant IAM permissions following least privilege for each component.
2. THE ShelfSentry_System SHALL NOT embed secrets in CloudFormation templates, application code, the Frontend, or version control.
3. THE Infrastructure_Stack SHALL define deletion policies and retention policies for stateful resources.

### Requirement 21: Infrastructure Teardown

**User Story:** As an operator, I want a documented teardown path, so that I can remove deployed resources without leaving orphaned infrastructure.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL provide a teardown guide describing how to remove the Infrastructure_Stack.
2. WHEN the Infrastructure_Stack is torn down following the teardown guide, THE Infrastructure_Stack SHALL remove the provisioned resources according to the defined deletion and retention policies.

### Requirement 22: Vision Testing

**User Story:** As a developer, I want a vision test suite covering normal and adverse conditions, so that vision behavior is verified before release.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL include vision tests covering valid, invalid, empty, and corrupt inputs.
2. THE ShelfSentry_System SHALL include vision tests covering varied lighting, shadows, dividers, occlusions, and multiple regions.
3. THE ShelfSentry_System SHALL include vision tests covering video frame extraction.
4. THE ShelfSentry_System SHALL include a vision test verifying that annotation coordinates are consistent across repeated runs of the same input.

### Requirement 23: Agent Testing

**User Story:** As a developer, I want an agent test suite covering causal behavior and safety, so that agent decision-making is verified before release.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL include an agent test verifying that different Vision_Finding inputs produce different next actions.
2. THE ShelfSentry_System SHALL include an agent test verifying that an inspect_crop action consumes the Candidate_Region identified by the preceding finding.
3. THE ShelfSentry_System SHALL include an agent test verifying that an ambiguous finding leads the Agent_Loop to gather more evidence.
4. THE ShelfSentry_System SHALL include an agent test verifying that conflicting evidence leads the Agent_Loop to select request_human_review.
5. THE ShelfSentry_System SHALL include an agent test verifying that the configured maximum iteration count is respected.
6. THE ShelfSentry_System SHALL include an agent test verifying that invalid tool arguments are rejected.
7. THE ShelfSentry_System SHALL include an agent test verifying that a model error causes the Fallback_Policy to be used.
8. THE ShelfSentry_System SHALL include an agent test verifying that the recorded Decision_Trace matches the tool calls that were made.

### Requirement 24: API and Infrastructure Testing

**User Story:** As a developer, I want API and infrastructure tests, so that request handling and infrastructure security are verified before release.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL include API tests covering request validation and rejection of unauthorized or invalid requests.
2. THE ShelfSentry_System SHALL include API tests covering missing S3 objects, failed jobs, and result retrieval.
3. THE ShelfSentry_System SHALL include a test that validates the CloudFormation template.
4. THE ShelfSentry_System SHALL include tests verifying IAM and storage security constraints.

### Requirement 25: Evaluation Set and Honest Reporting

**User Story:** As a developer, I want a reproducible labeled evaluation set and honest reporting, so that reported performance reflects measured results only.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL include a small reproducible labeled evaluation set.
2. WHEN the evaluation set is run, THE ShelfSentry_System SHALL produce Evaluation_Result records containing measured results including false positives and false negatives.
3. THE ShelfSentry_System SHALL report documented limitations alongside Evaluation_Result records.
4. THE ShelfSentry_System SHALL report only measured results in Evaluation_Result records and SHALL NOT report fabricated metrics or confidence scores.

### Requirement 26: Documentation and Submission Artifacts

**User Story:** As a reviewer, I want complete documentation and submission artifacts, so that I can set up, understand, evaluate, and deploy the system.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL provide a README covering setup, architecture, testing, deployment, and cleanup.
2. THE ShelfSentry_System SHALL provide docs/architecture.md containing a sequence diagram covering the flow from OpenCV analysis to agent to MCP tool to new evidence to final decision.
3. THE ShelfSentry_System SHALL provide docs/evaluation.md, docs/limitations.md, and docs/security.md.
4. THE ShelfSentry_System SHALL pin dependency versions and SHALL document the exact OpenCV 5 version and where it performs substantive analysis.
5. THE ShelfSentry_System SHALL include licensed sample inputs, a 5-minute demo script, a technical report, and evidence of the visual-to-action loop including screenshots or recordings.

### Requirement 27: Honesty and Safety Guardrails

**User Story:** As a stakeholder, I want strict honesty and safety guardrails, so that the system does not mislead users or destroy data.

#### Acceptance Criteria

1. THE ShelfSentry_System SHALL present a feature, test, or deployment as working only after that feature, test, or deployment has been verified.
2. THE ShelfSentry_System SHALL NOT fabricate metrics or confidence scores in any output.
3. WHEN user data is deleted, THE ShelfSentry_System SHALL perform the deletion through an explicit deletion or retention action and SHALL NOT delete user data silently.
4. WHEN the ShelfSentry_System reports that a Candidate_Region appears empty, THE ShelfSentry_System SHALL include the documented visual caveats covering shadow, occlusion, dividers, and image quality rather than asserting an out-of-stock condition as proven.
