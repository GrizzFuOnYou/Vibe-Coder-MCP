import fs from 'fs-extra';
import path from 'path';
import { z } from 'zod';
import { CallToolResult, McpError, ErrorCode } from '@modelcontextprotocol/sdk/types.js'; // Added McpError, ErrorCode
import { OpenRouterConfig } from '../../types/workflow.js';
import { performDirectLlmCall } from '../../utils/llmHelper.js'; // Import the new helper
import { performResearchQuery } from '../../utils/researchHelper.js';
import logger from '../../logger.js';
import { registerTool, ToolDefinition, ToolExecutor, ToolExecutionContext } from '../../services/routing/toolRegistry.js'; // Import ToolExecutionContext
import { AppError, ApiError, ConfigurationError, ToolExecutionError } from '../../utils/errors.js'; // Import necessary errors
import { jobManager, JobStatus } from '../../services/job-manager/index.js'; // Import job manager & status
import { sseNotifier } from '../../services/sse-notifier/index.js'; // Import SSE notifier

// User stories generator-specific system prompt
const USER_STORIES_SYSTEM_PROMPT = `
# User Stories Generator - Using Research Context

# ROLE & GOAL
You are an expert Agile Business Analyst and Product Owner AI assistant. Your goal is to generate a comprehensive and well-structured set of User Stories, including Epics and Acceptance Criteria, in Markdown format.

# CORE TASK
Generate detailed user stories based on the user's product description and the provided research context.

# INPUT HANDLING
- Analyze the 'productDescription' to understand the product's purpose, core features, and intended value.
- You will also receive 'Pre-Generation Research Context'.

# RESEARCH CONTEXT INTEGRATION
- **CRITICAL:** Carefully review the '## Pre-Generation Research Context (From Perplexity Sonar Deep Research)' section provided in the user prompt.
- This section contains insights on: User Personas & Stakeholders, User Workflows & Use Cases, and User Experience Expectations & Pain Points.
- **Use these insights** heavily to:
    - Define realistic 'As a [user type/persona]' roles based on the research.
    - Create stories that address identified 'User Workflows & Use Cases'.
    - Ensure stories align with 'User Experience Expectations' and address 'Pain Points'.
    - Inform the 'Priority' and 'Value/Benefit' parts of the stories.
- **Synthesize**, don't just list research findings. Create user stories that *embody* the research.

# OUTPUT FORMAT & STRUCTURE (Strict Markdown)
- Your entire response **MUST** be valid Markdown.
- Start **directly** with the main title: '# User Stories: [Inferred Product Name]'
- Organize stories hierarchically using Markdown headings:
    - \`## Epic: [Epic Title]\` (e.g., \`## Epic: User Authentication\`)
    - \`### User Story: [Story Title]\` (e.g., \`### User Story: User Registration\`)
- For **each User Story**, use the following precise template within its \`###\` section:

  **ID:** US-[auto-incrementing number, e.g., US-101]
  **Title:** [Concise Story Title]
  
  **Story:**
  > As a **[User Role/Persona - informed by research]**,
  > I want to **[perform an action or achieve a goal]**
  > So that **[I gain a specific benefit - linked to user needs/pain points from research]**.
  
  **Acceptance Criteria:**
  *   GIVEN [precondition/context] WHEN [action is performed] THEN [expected, testable outcome].
  *   GIVEN [another context] WHEN [different action] THEN [another outcome].
  *   *(Provide multiple, specific, measurable criteria)*
  
  **Priority:** [High | Medium | Low - informed by perceived value/dependencies/research]
  **Dependencies:** [List of User Story IDs this depends on, e.g., US-100 | None]
  **(Optional) Notes:** [Any clarifying details or technical considerations.]

# QUALITY ATTRIBUTES
- **INVEST Principles:** Ensure stories are Independent, Negotiable, Valuable, Estimable, Small (appropriately sized), and Testable (via Acceptance Criteria).
- **User-Centric:** Focus on user roles, actions, and benefits, informed by research personas and needs.
- **Clear Acceptance Criteria:** Criteria must be specific, unambiguous, and testable.
- **Comprehensive:** Cover the core functionality implied by the description and research workflows.
- **Well-Structured:** Adhere strictly to the Epic/Story hierarchy and template format.
- **Consistent:** Use consistent terminology and formatting.

# CONSTRAINTS (Do NOT Do the Following)
- **NO Conversational Filler:** Start directly with the '# User Stories: ...' title. No intros, summaries, or closings.
- **NO Markdown Violations:** Strictly adhere to the specified Markdown format (headings, blockquotes for the story, lists for AC).
- **NO Implementation Details:** Focus on *what* the user needs, not *how* it will be built (unless specified in 'Notes').
- **NO External Knowledge:** Base stories *only* on the provided inputs and research context.
- **NO Process Commentary:** Do not mention the research process in the output.
- **Strict Formatting:** Use \`##\` for Epics, \`###\` for Stories. Use the exact field names (ID, Title, Story, Acceptance Criteria, etc.) in bold. Use Markdown blockquotes for the As a/I want/So that structure.
`;

// Define Input Type based on Schema
const userStoriesInputSchemaShape = {
  productDescription: z.string().min(10, { message: "Product description must be at least 10 characters." }).describe("Description of the product to create user stories for")
};

/**
 * Generate user stories based on a product description.
 * This function now acts as the executor for the 'generate-user-stories' tool.
 * @param params The validated tool parameters.
 * @param config OpenRouter configuration.
 * @returns A Promise resolving to a CallToolResult object.
 */
export const generateUserStories: ToolExecutor = async (
  params: Record<string, unknown>, // More type-safe than 'any'
  config: OpenRouterConfig,
  context?: ToolExecutionContext // Add context parameter
): Promise<CallToolResult> => { // Return CallToolResult
  // ---> Step 2.5(US).2: Inject Dependencies & Get Session ID <---
  const sessionId = context?.sessionId || 'unknown-session';
  if (sessionId === 'unknown-session') {
      logger.warn({ tool: 'generateUserStories' }, 'Executing tool without a valid sessionId. SSE progress updates will not be sent.');
  }

  // Log the config received by the executor
  logger.debug({
    configReceived: true,
    hasLlmMapping: Boolean(config.llm_mapping),
    mappingKeys: config.llm_mapping ? Object.keys(config.llm_mapping) : []
  }, 'generateUserStories executor received config');

  const productDescription = params.productDescription as string; // Assert type after validation

  // ---> Step 2.5(US).3: Create Job & Return Job ID <---
  const jobId = jobManager.createJob('generate-user-stories', params);
  logger.info({ jobId, tool: 'generateUserStories', sessionId }, 'Starting background job.');

  // Return immediately
  const initialResponse: CallToolResult = {
    content: [{ type: 'text', text: `User stories generation started. Job ID: ${jobId}` }],
    isError: false,
  };

  // ---> Step 2.5(US).4: Wrap Logic in Async Block <---
  setImmediate(async () => {
    const logs: string[] = []; // Keep logs specific to this job execution

    // ---> Step 2.5(US).7: Update Final Result/Error Handling (Try Block Start) <---
    try {
      // ---> Step 2.5(US).6: Add Progress Updates (Initial) <---
      jobManager.updateJobStatus(jobId, JobStatus.RUNNING, 'Starting user stories generation process...');
      sseNotifier.sendProgress(sessionId, jobId, JobStatus.RUNNING, 'Starting user stories generation process...');
      logs.push(`[${new Date().toISOString()}] Starting user stories generation for: ${productDescription.substring(0, 50)}...`);

      // ---> Step 2.5(US).6: Add Progress Updates (Research Start) <---
      logger.info({ jobId, inputs: { productDescription: productDescription.substring(0, 50) } }, "User Stories Generator: Starting pre-generation research...");
      jobManager.updateJobStatus(jobId, JobStatus.RUNNING, 'Performing pre-generation research...');
      sseNotifier.sendProgress(sessionId, jobId, JobStatus.RUNNING, 'Performing pre-generation research...');
      logs.push(`[${new Date().toISOString()}] Starting pre-generation research.`);

      let researchContext = '';
    try {
      // Define relevant research queries
      const query1 = `User personas and stakeholders for: ${productDescription}`;
      const query2 = `Common user workflows and use cases for: ${productDescription}`;
      const query3 = `User experience expectations and pain points for: ${productDescription}`;
      
      // Execute research queries in parallel using Perplexity
      const researchResults = await Promise.allSettled([
        performResearchQuery(query1, config), // Uses config.perplexityModel (perplexity/sonar-deep-research)
        performResearchQuery(query2, config),
        performResearchQuery(query3, config)
      ]);
      
      // Process research results
      researchContext = "## Pre-Generation Research Context (From Perplexity Sonar Deep Research):\n\n";
      
      // Add results that were fulfilled
      researchResults.forEach((result, index) => {
        const queryLabels = ["User Personas & Stakeholders", "User Workflows & Use Cases", "User Experience Expectations & Pain Points"];
        if (result.status === "fulfilled") {
          researchContext += `### ${queryLabels[index]}:\n${result.value.trim()}\n\n`;
        } else {
          logger.warn({ error: result.reason }, `Research query ${index + 1} failed`);
          researchContext += `### ${queryLabels[index]}:\n*Research on this topic failed.*\n\n`;
        }
      });

      // ---> Step 2.5(US).6: Add Progress Updates (Research End) <---
      logger.info({ jobId }, "User Stories Generator: Pre-generation research completed.");
      jobManager.updateJobStatus(jobId, JobStatus.RUNNING, 'Research complete. Starting main user stories generation...');
      sseNotifier.sendProgress(sessionId, jobId, JobStatus.RUNNING, 'Research complete. Starting main user stories generation...');
      logs.push(`[${new Date().toISOString()}] Pre-generation research completed.`);

    } catch (researchError) {
      logger.error({ jobId, err: researchError }, "User Stories Generator: Error during research aggregation");
      logs.push(`[${new Date().toISOString()}] Error during research aggregation: ${researchError instanceof Error ? researchError.message : String(researchError)}`);
      // Include error in context but continue
      researchContext = "## Pre-Generation Research Context:\n*Error occurred during research phase.*\n\n";
      sseNotifier.sendProgress(sessionId, jobId, JobStatus.RUNNING, 'Warning: Error during research phase. Continuing generation...');
    }

    // Create the main generation prompt with combined research and inputs
    const mainGenerationPrompt = `Create comprehensive user stories for the following product:\n\n${productDescription}\n\n${researchContext}`;

    // ---> Step 2.5(US).6: Add Progress Updates (LLM Call Start) <---
    logger.info({ jobId }, "User Stories Generator: Starting main generation using direct LLM call...");
    jobManager.updateJobStatus(jobId, JobStatus.RUNNING, 'Generating user stories content via LLM...');
    sseNotifier.sendProgress(sessionId, jobId, JobStatus.RUNNING, 'Generating user stories content via LLM...');
    logs.push(`[${new Date().toISOString()}] Calling LLM for main user stories generation.`);

    const userStoriesMarkdown = await performDirectLlmCall(
      mainGenerationPrompt,
      USER_STORIES_SYSTEM_PROMPT, // Pass the system prompt
      config,
      'user_stories_generation', // Logical task name
      0.3 // Slightly higher temp might be okay for creative text like stories
    );

    // ---> Step 2.5(US).6: Add Progress Updates (LLM Call End) <---
    logger.info({ jobId }, "User Stories Generator: Main generation completed.");
    jobManager.updateJobStatus(jobId, JobStatus.RUNNING, 'Processing LLM response...');
    sseNotifier.sendProgress(sessionId, jobId, JobStatus.RUNNING, 'Processing LLM response...');
    logs.push(`[${new Date().toISOString()}] Received response from LLM.`);

    // Basic validation: Check if the output looks like Markdown and contains expected elements
    if (!userStoriesMarkdown || typeof userStoriesMarkdown !== 'string' || !userStoriesMarkdown.trim().startsWith('# User Stories:')) {
      logger.warn({ jobId, markdown: userStoriesMarkdown?.substring(0, 100) }, 'User stories generation returned empty or potentially invalid Markdown format.');
      logs.push(`[${new Date().toISOString()}] Validation Error: LLM output invalid format.`);
      throw new ToolExecutionError('User stories generation returned empty or invalid Markdown content.');
    }

    // Format the user stories (already should be formatted by LLM, just add timestamp)
    const formattedResult = `${userStoriesMarkdown}\n\n_Generated: ${new Date().toLocaleString()}_`;

    // ---> Step 2.5(US).7: Update Final Result/Error Handling (Set Success Result) <---
    const finalResult: CallToolResult = {
      // Include file path in success message
      content: [{ type: "text", text: `User stories generated successfully.\n\n${formattedResult}` }],
      isError: false
    };
    jobManager.setJobResult(jobId, finalResult);
    // Optional explicit SSE: sseNotifier.sendProgress(sessionId, jobId, JobStatus.COMPLETED, 'User stories generation completed successfully.');

    // ---> Step 2.5(US).7: Update Final Result/Error Handling (Catch Block) <---
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger.error({ err: error, jobId, tool: 'generate-user-stories', params }, `User Stories Generator Error: ${errorMsg}`);
      logs.push(`[${new Date().toISOString()}] Error: ${errorMsg}`);

      // Handle specific errors from direct call or research
      let appError: AppError;
      const cause = error instanceof Error ? error : undefined;
      if (error instanceof AppError) {
        appError = error;
      } else {
        appError = new ToolExecutionError(`Failed to generate user stories: ${errorMsg}`, { params }, cause);
      }

      const mcpError = new McpError(ErrorCode.InternalError, appError.message, appError.context);
      const errorResult: CallToolResult = {
        content: [{ type: 'text', text: `Error during background job ${jobId}: ${mcpError.message}\n\nLogs:\n${logs.join('\n')}` }],
        isError: true,
        errorDetails: mcpError
      };

      // Store error result in Job Manager
      jobManager.setJobResult(jobId, errorResult);
      // Send final failed status via SSE (optional if jobManager handles it)
      sseNotifier.sendProgress(sessionId, jobId, JobStatus.FAILED, `Job failed: ${mcpError.message}`);
    }
  }); // ---> END OF setImmediate WRAPPER <---

  return initialResponse; // Return the initial response with Job ID
};

// --- Tool Registration ---

// Tool definition for the user stories generator tool
const userStoriesToolDefinition: ToolDefinition = {
  name: "generate-user-stories",
  description: "Creates detailed user stories with acceptance criteria based on a product description and research.",
  inputSchema: userStoriesInputSchemaShape, // Use the raw shape
  executor: generateUserStories // Reference the adapted function
};

// Register the tool with the central registry
registerTool(userStoriesToolDefinition);
