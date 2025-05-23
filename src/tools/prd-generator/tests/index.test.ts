// src/tools/prd-generator/tests/index.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'; // Keep only one import
import { generatePRD, PRD_SYSTEM_PROMPT, initDirectories } from '../index.js'; // Keep only one import
import { OpenRouterConfig } from '../../../types/workflow.js';
import * as researchHelper from '../../../utils/researchHelper.js';
import * as llmHelper from '../../../utils/llmHelper.js';
import { ApiError, AppError, ToolExecutionError } from '../../../utils/errors.js';
import fs from 'fs-extra';
import logger from '../../../logger.js';
import { jobManager, JobStatus } from '../../../services/job-manager/index.js'; // Import Job Manager
import { sseNotifier } from '../../../services/sse-notifier/index.js'; // Import SSE Notifier
import { CallToolResult } from '@modelcontextprotocol/sdk/types.js'; // Import CallToolResult

// Mock dependencies
vi.mock('../../../utils/researchHelper.js');
vi.mock('../../../utils/llmHelper.js'); // Mock the new helper
vi.mock('fs-extra');
vi.mock('../../../logger.js');
vi.mock('../../../services/job-manager/index.js'); // Mock Job Manager
vi.mock('../../../services/sse-notifier/index.js'); // Mock SSE Notifier

// Define helper variables for mocks using vi.mocked() for better type handling
const mockPerformResearchQuery = vi.mocked(researchHelper.performResearchQuery);
const mockPerformDirectLlmCall = vi.mocked(llmHelper.performDirectLlmCall); // Mock the new helper
const mockWriteFile = vi.mocked(fs.writeFile);
const mockEnsureDir = vi.mocked(fs.ensureDir);

// Helper to advance timers and allow setImmediate to run
const runAsyncTicks = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await vi.advanceTimersToNextTimerAsync(); // Allow setImmediate/promises to resolve
  }
};

const mockJobId = 'mock-prd-job-id';

describe('PRD Generator Tool Executor (Async)', () => {
  // Mock data and responses
  const mockConfig: OpenRouterConfig = {
    baseUrl: 'mock-url',
    apiKey: 'test-api-key',
    geminiModel: 'google/gemini-2.5-pro-exp-03-25:free',
    perplexityModel: 'perplexity/sonar-deep-research'
  };
  const mockContext = { sessionId: 'test-session-prd' };

  const mockResearchResults = [
    "Mock market analysis research data",
    "Mock user needs research data",
    "Mock industry standards research data"
  ];

  const mockGeneratedPRD = "# Mock PRD\n\nThis is a mock PRD generated by the test.";

  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();

    // Default mocks for successful execution
    mockEnsureDir.mockResolvedValue(undefined);
    mockWriteFile.mockResolvedValue(undefined);
    // Setup mock chain for research queries
    mockPerformResearchQuery
        .mockResolvedValueOnce(mockResearchResults[0]) // Market Analysis
        .mockResolvedValueOnce(mockResearchResults[1]) // User Needs
        .mockResolvedValueOnce(mockResearchResults[2]); // Standards
    mockPerformDirectLlmCall.mockResolvedValue(mockGeneratedPRD); // Mock the direct call

    // Mock Job Manager methods
    vi.mocked(jobManager.createJob).mockReturnValue(mockJobId);
    vi.mocked(jobManager.updateJobStatus).mockReturnValue(true);
    vi.mocked(jobManager.setJobResult).mockReturnValue(true);

    // Enable fake timers
    vi.useFakeTimers();
  });

   afterEach(() => {
       vi.restoreAllMocks();
       vi.useRealTimers(); // Restore real timers
   });

  it('should return job ID and complete successfully in background', async () => {
    const productDescription = "Fancy New Widget";
    const params = { productDescription };

    // --- Initial Call ---
    const initialResult = await generatePRD(params, mockConfig, mockContext);
    expect(initialResult.isError).toBe(false);
    expect(initialResult.content[0]?.text).toContain(`PRD generation started. Job ID: ${mockJobId}`);
    expect(jobManager.createJob).toHaveBeenCalledWith('generate-prd', params);

    // Verify underlying logic not called yet
    expect(mockPerformResearchQuery).not.toHaveBeenCalled();
    expect(mockPerformDirectLlmCall).not.toHaveBeenCalled();
    expect(mockWriteFile).not.toHaveBeenCalled();
    expect(jobManager.setJobResult).not.toHaveBeenCalled();

    // --- Advance Timers ---
    await runAsyncTicks(5); // Allow for research + generation

    // --- Verify Async Operations ---
    // 1. Verify Research
    expect(mockPerformResearchQuery).toHaveBeenCalledTimes(3);
    expect(mockPerformResearchQuery).toHaveBeenCalledWith(expect.stringContaining(`Market analysis`), mockConfig);

    // 2. Verify LLM Call
    expect(mockPerformDirectLlmCall).toHaveBeenCalledTimes(1);
    const llmCallArgs = mockPerformDirectLlmCall.mock.calls[0];
    expect(llmCallArgs[0]).toContain(productDescription); // Main prompt check
    expect(llmCallArgs[1]).toBe(PRD_SYSTEM_PROMPT); // System prompt check
    expect(llmCallArgs[3]).toBe('prd_generation'); // Logical task name check

    // 3. Verify File Saving
    expect(mockWriteFile).toHaveBeenCalledTimes(1);
    const writeFileArgs = mockWriteFile.mock.calls[0];
    expect(writeFileArgs[0]).toMatch(/prd-generator[\\/].*fancy-new-widget.*-prd\.md$/); // Path check
    expect(writeFileArgs[1]).toContain(mockGeneratedPRD); // Content check

    // 4. Verify Final Job Result
    expect(jobManager.setJobResult).toHaveBeenCalledTimes(1);
    const finalResultArgs = vi.mocked(jobManager.setJobResult).mock.calls[0];
    expect(finalResultArgs[0]).toBe(mockJobId);
    expect(finalResultArgs[1].isError).toBe(false);
    expect(finalResultArgs[1].content[0]?.text).toContain(`PRD generated successfully and saved to:`);
    expect(finalResultArgs[1].content[0]?.text).toContain(mockGeneratedPRD);

    // 5. Verify SSE Calls (basic)
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Starting PRD generation'));
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Performing pre-generation research'));
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Generating PRD content via LLM'));
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Saving PRD to file'));
  });

  it('should handle research failures gracefully (async)', async () => {
    // Mock one research query to fail
    mockPerformResearchQuery.mockReset();
    mockPerformResearchQuery
        .mockRejectedValueOnce(new ApiError('Market research failed', 500)) // Market Analysis fails
        .mockResolvedValueOnce(mockResearchResults[1]) // User Needs succeeds
        .mockResolvedValueOnce(mockResearchResults[2]); // Standards succeeds

    const productDescription = "Widget With Failing Research";
    // --- Initial Call ---
    await generatePRD({ productDescription }, mockConfig, mockContext);
    // --- Advance Timers ---
    await runAsyncTicks(5);
    // --- Verify Async Operations ---
    expect(mockPerformDirectLlmCall).toHaveBeenCalledTimes(1); // Generation should still run
    const mainPromptArg = mockPerformDirectLlmCall.mock.calls[0][0];
    expect(mainPromptArg).toContain("### Market Analysis:\n*Research on this topic failed.*\n\n"); // Check failure message in prompt
    expect(mockWriteFile).toHaveBeenCalledTimes(1); // File should still be saved
    expect(jobManager.setJobResult).toHaveBeenCalledTimes(1); // Job should complete successfully
    expect(vi.mocked(jobManager.setJobResult).mock.calls[0][1].isError).toBe(false);
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Warning: Error during research phase'));
  });

  it('should set job to FAILED if direct LLM call throws error (async)', async () => {
       const llmError = new ApiError("LLM call failed", 500);
       mockPerformDirectLlmCall.mockRejectedValueOnce(llmError); // Throw error

       const productDescription = "Confusing Widget";
       // --- Initial Call ---
       await generatePRD({ productDescription }, mockConfig, mockContext);
       // --- Advance Timers ---
       await runAsyncTicks(5);
       // --- Verify Async Operations ---
       expect(mockPerformResearchQuery).toHaveBeenCalledTimes(3); // Research should have run
       expect(mockPerformDirectLlmCall).toHaveBeenCalledTimes(1); // LLM call was attempted
       expect(mockWriteFile).not.toHaveBeenCalled(); // File shouldn't be saved
       expect(jobManager.setJobResult).toHaveBeenCalledTimes(1); // Job should fail
       const finalResultArgs = vi.mocked(jobManager.setJobResult).mock.calls[0];
       expect(finalResultArgs[0]).toBe(mockJobId);
       expect(finalResultArgs[1].isError).toBe(true);
       expect(finalResultArgs[1].content[0]?.text).toContain('Error during background job');
       const errorDetails = finalResultArgs[1].errorDetails as any;
       expect(errorDetails?.message).toContain('Failed to generate PRD: LLM call failed'); // Check wrapped message
   });

   it('should set job to FAILED if file writing fails (async)', async () => {
       const fileWriteError = new Error("Disk full");
       mockWriteFile.mockRejectedValueOnce(fileWriteError);

       const productDescription = "Unsavable Widget";
       // --- Initial Call ---
       await generatePRD({ productDescription }, mockConfig, mockContext);
       // --- Advance Timers ---
       await runAsyncTicks(5);
       // --- Verify Async Operations ---
       expect(mockPerformResearchQuery).toHaveBeenCalledTimes(3);
       expect(mockPerformDirectLlmCall).toHaveBeenCalledTimes(1);
       expect(mockWriteFile).toHaveBeenCalledTimes(1); // Write was attempted
       expect(jobManager.setJobResult).toHaveBeenCalledTimes(1); // Job should fail
       const finalResultArgs = vi.mocked(jobManager.setJobResult).mock.calls[0];
       expect(finalResultArgs[0]).toBe(mockJobId);
       expect(finalResultArgs[1].isError).toBe(true);
       expect(finalResultArgs[1].content[0]?.text).toContain('Error during background job');
       const errorDetails = finalResultArgs[1].errorDetails as any;
       expect(errorDetails?.message).toContain('Failed to generate PRD: Disk full');
  });

  // --- Snapshot Test (Adapted for Async) ---
  it('should set final job result content matching snapshot (async)', async () => {
      const productDescription = "A sample product for snapshot";
      const params = { productDescription };
      const consistentMockPRD = "# Mock PRD\n## Section 1\nDetails...\n## Section 2\nMore details...";

      // Reset mocks for consistency
      mockPerformResearchQuery.mockReset();
      mockPerformResearchQuery.mockResolvedValue("Consistent mock research.");
      mockPerformDirectLlmCall.mockReset();
      mockPerformDirectLlmCall.mockResolvedValue(consistentMockPRD); // Mock direct call
      mockWriteFile.mockReset();
      mockWriteFile.mockResolvedValue(undefined); // Mock successful write

      // --- Initial Call ---
      await generatePRD(params, mockConfig, mockContext);
      // --- Advance Timers ---
      await runAsyncTicks(5);
      // --- Verify Async Operations ---
      expect(jobManager.setJobResult).toHaveBeenCalledTimes(1);
      const finalResultArgs = vi.mocked(jobManager.setJobResult).mock.calls[0];
      expect(finalResultArgs[1].isError).toBe(false);
      const finalResult = finalResultArgs[1] as CallToolResult;

      // Snapshot the main content excluding the timestamp and file path
      const resultText = finalResult.content?.[0]?.text ?? '';
      const contentToSnapshot = (resultText as string)
          .replace(/_Generated: .*_$/, '') // Remove timestamp
          .replace(/PRD generated successfully and saved to: .*?-prd\.md\n\n/, '') // Remove file path line
          .trim();
      expect(contentToSnapshot).toMatchSnapshot('PRD Generator Content');

      // Verify file write happened
      expect(mockWriteFile).toHaveBeenCalledTimes(1);
      expect(mockWriteFile).toHaveBeenCalledWith(expect.stringContaining('-prd.md'), expect.any(String), 'utf8');
  });
});
