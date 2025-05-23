// src/tools/rules-generator/tests/index.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'; // Keep only one import
import { generateRules } from '../index.js';
import { OpenRouterConfig } from '../../../types/workflow.js';
import * as researchHelper from '../../../utils/researchHelper.js';
import * as llmHelper from '../../../utils/llmHelper.js'; // Import the new helper
import fs from 'fs-extra';
import { jobManager, JobStatus } from '../../../services/job-manager/index.js'; // Import Job Manager
import { sseNotifier } from '../../../services/sse-notifier/index.js'; // Import SSE Notifier
import { CallToolResult } from '@modelcontextprotocol/sdk/types.js'; // Import CallToolResult
import { ApiError, ToolExecutionError } from '../../../utils/errors.js'; // Import necessary errors
import logger from '../../../logger.js'; // Import logger

// Mock dependencies
vi.mock('../../../utils/researchHelper.js');
vi.mock('../../../utils/llmHelper.js'); // Mock the new helper
vi.mock('fs-extra');
vi.mock('../../../services/job-manager/index.js'); // Mock Job Manager
vi.mock('../../../services/sse-notifier/index.js'); // Mock SSE Notifier
vi.mock('../../../logger.js'); // Mock logger

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

const mockJobId = 'mock-rules-job-id';

describe('Rules Generator (Async)', () => {
  // Mock data and responses
  const mockConfig: OpenRouterConfig = {
    baseUrl: 'https://api.example.com',
    apiKey: 'test-api-key',
    geminiModel: 'google/gemini-2.5-pro-exp-03-25:free',
    perplexityModel: 'perplexity/sonar-deep-research'
  };
  const mockContext = { sessionId: 'test-session-rules' };

  const mockUserStories = 'US-001: As a user, I want to...';
  const mockRuleCategories = ['Code Style', 'Architecture', 'Security'];

  const mockResearchResults = [
    "Mock best practices research data",
    "Mock rule categories research data",
    "Mock architecture patterns research data"
  ];

  const mockGeneratedRules = "# Mock Rules\n\nThis is a mock rules document generated by the test.";

  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();

    // Default mocks for successful execution
    mockEnsureDir.mockResolvedValue(undefined);
    mockWriteFile.mockResolvedValue(undefined);
    // Setup mock chain for research queries
    mockPerformResearchQuery
        .mockResolvedValueOnce(mockResearchResults[0]) // Practices
        .mockResolvedValueOnce(mockResearchResults[1]) // Categories
        .mockResolvedValueOnce(mockResearchResults[2]); // Architecture
    mockPerformDirectLlmCall.mockResolvedValue(mockGeneratedRules); // Mock the direct call

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
    const params = {
      productDescription: "A test product description",
      userStories: mockUserStories,
      ruleCategories: mockRuleCategories
    };

    // --- Initial Call ---
    const initialResult = await generateRules(params, mockConfig, mockContext);
    expect(initialResult.isError).toBe(false);
    expect(initialResult.content[0]?.text).toContain(`Development rules generation started. Job ID: ${mockJobId}`);
    expect(jobManager.createJob).toHaveBeenCalledWith('generate-rules', params);

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
    const researchCalls = vi.mocked(researchHelper.performResearchQuery).mock.calls;
    expect(researchCalls[0][0]).toContain('practices');
    expect(researchCalls[1][0]).toContain('categories');
    expect(researchCalls[2][0]).toContain('architecture');

    // 2. Verify LLM Call
    expect(mockPerformDirectLlmCall).toHaveBeenCalledTimes(1);
    const llmCallArgs = mockPerformDirectLlmCall.mock.calls[0];
    expect(llmCallArgs[0]).toContain(params.productDescription); // Main prompt check
    expect(llmCallArgs[1]).toContain("# Rules Generator"); // System prompt check
    expect(llmCallArgs[3]).toBe('rules_generation'); // Logical task name check

    // 3. Verify File Saving
    expect(mockWriteFile).toHaveBeenCalledTimes(1);
    const writeFileArgs = mockWriteFile.mock.calls[0];
    expect(writeFileArgs[0]).toMatch(/rules-generator[\\/].*a-test-product-description.*-rules\.md$/); // Path check
    expect(writeFileArgs[1]).toContain(mockGeneratedRules); // Content check

    // 4. Verify Final Job Result
    expect(jobManager.setJobResult).toHaveBeenCalledTimes(1);
    const finalResultArgs = vi.mocked(jobManager.setJobResult).mock.calls[0];
    expect(finalResultArgs[0]).toBe(mockJobId);
    expect(finalResultArgs[1].isError).toBe(false);
    expect(finalResultArgs[1].content[0]?.text).toContain(`Development rules generated successfully and saved to:`);
    expect(finalResultArgs[1].content[0]?.text).toContain(mockGeneratedRules);

    // 5. Verify SSE Calls (basic)
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Starting rules generation'));
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Performing pre-generation research'));
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Generating rules content via LLM'));
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Saving rules to file'));
  });

  it('should handle custom rule categories in the research process (async)', async () => {
    const params = {
      productDescription: "Test product",
      ruleCategories: ["Security", "Performance"]
    };
    // --- Initial Call ---
    await generateRules(params, mockConfig, mockContext);
    // --- Advance Timers ---
    await runAsyncTicks(5);
    // --- Verify Async Operations ---
    expect(mockPerformResearchQuery).toHaveBeenCalledTimes(3);
    const secondResearchQuery = vi.mocked(researchHelper.performResearchQuery).mock.calls[1][0];
    expect(secondResearchQuery).toContain("Security, Performance");
    expect(jobManager.setJobResult).toHaveBeenCalledTimes(1); // Should still complete
    expect(vi.mocked(jobManager.setJobResult).mock.calls[0][1].isError).toBe(false);
  });

  it('should handle research failures gracefully (async)', async () => {
    // Mock a failed research query
    vi.mocked(Promise.allSettled).mockResolvedValueOnce([
      { status: 'rejected', reason: new Error('Research failed') },
      { status: 'fulfilled', value: mockResearchResults[1] },
      { status: 'fulfilled', value: mockResearchResults[2] }
    ]);
    const params = {
      productDescription: "A test product description",
      userStories: mockUserStories,
      ruleCategories: mockRuleCategories
    };
    // --- Initial Call ---
    await generateRules(params, mockConfig, mockContext);
    // --- Advance Timers ---
    await runAsyncTicks(5);
    // --- Verify Async Operations ---
    expect(mockPerformDirectLlmCall).toHaveBeenCalledTimes(1); // Generation should still run
    const generationPrompt = vi.mocked(llmHelper.performDirectLlmCall).mock.calls[0][0];
    expect(generationPrompt).toContain("### Best Practices:\n*Research on this topic failed.*\n\n"); // Check failure message
    expect(mockWriteFile).toHaveBeenCalledTimes(1); // File should still be saved
    expect(jobManager.setJobResult).toHaveBeenCalledTimes(1); // Job should complete successfully
    expect(vi.mocked(jobManager.setJobResult).mock.calls[0][1].isError).toBe(false);
    expect(sseNotifier.sendProgress).toHaveBeenCalledWith(mockContext.sessionId, mockJobId, JobStatus.RUNNING, expect.stringContaining('Warning: Error during research phase'));
  });

  it('should set job to FAILED if direct LLM call throws error (async)', async () => {
       const llmError = new ApiError("LLM call failed", 500);
       mockPerformDirectLlmCall.mockRejectedValueOnce(llmError); // Throw error
       const params = { productDescription: "Confusing Product" };
       // --- Initial Call ---
       await generateRules(params, mockConfig, mockContext);
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
       expect(errorDetails?.message).toContain('Failed to generate development rules: LLM call failed');
   });

   it('should set job to FAILED if file writing fails (async)', async () => {
       const fileWriteError = new Error("Disk full");
       mockWriteFile.mockRejectedValueOnce(fileWriteError);
       const params = { productDescription: "Unsavable Rules" };
       // --- Initial Call ---
       await generateRules(params, mockConfig, mockContext);
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
       expect(errorDetails?.message).toContain('Failed to generate development rules: Disk full');
  });

  // --- Snapshot Test (Adapted for Async) ---
  it('should set final job result content matching snapshot (async)', async () => {
      const productDescription = "A sample product for snapshot";
      const params = { productDescription };
      const consistentMockRules = "# Mock Rules\n## Category 1\nRule details...";

      // Reset mocks for consistency
      mockPerformResearchQuery.mockReset();
      mockPerformResearchQuery.mockResolvedValue("Consistent mock research.");
      mockPerformDirectLlmCall.mockReset();
      mockPerformDirectLlmCall.mockResolvedValue(consistentMockRules);
      mockWriteFile.mockReset();
      mockWriteFile.mockResolvedValue(undefined);

      // --- Initial Call ---
      await generateRules(params, mockConfig, mockContext);
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
          .replace(/Development rules generated successfully and saved to: .*?-rules\.md\n\n/, '') // Remove file path line
          .trim();
      expect(contentToSnapshot).toMatchSnapshot('Rules Generator Content');

      // Verify file write happened
      expect(mockWriteFile).toHaveBeenCalledTimes(1);
      expect(mockWriteFile).toHaveBeenCalledWith(expect.stringContaining('-rules.md'), expect.any(String), 'utf8');
  });
});
