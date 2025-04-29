import { randomUUID } from 'crypto';
import { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import logger from '../../logger.js';
// Import sseNotifier here later when it exists to notify on status/result changes
import { sseNotifier } from '../sse-notifier/index.js';

/**
 * Represents the possible statuses of a background job.
 */
export enum JobStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/**
 * Defines the structure of a background job tracked by the JobManager.
 */
export interface Job {
  id: string;
  toolName: string;
  params: Record<string, unknown>;
  status: JobStatus;
  createdAt: number;
  updatedAt: number;
  progressMessage?: string; // Optional message describing the current step
  result?: CallToolResult; // Final result (success or error)
}

/**
 * Manages the state of background jobs.
 * Uses a Singleton pattern.
 */
class JobManager {
  private jobs = new Map<string, Job>();

  /**
   * Creates a new job and stores it.
   * @param toolName The name of the tool being executed.
   * @param params The parameters the tool was called with.
   * @returns The ID of the newly created job.
   */
  createJob(toolName: string, params: Record<string, unknown>): string {
    const jobId = randomUUID();
    const now = Date.now();
    const newJob: Job = {
      id: jobId,
      toolName,
      params,
      status: JobStatus.PENDING,
      createdAt: now,
      updatedAt: now,
    };
    this.jobs.set(jobId, newJob);
    logger.info({ jobId, toolName }, `Created new background job.`);
    sseNotifier.sendProgress('auto-session', jobId, JobStatus.PENDING, 'Job created');
    return jobId;
  }

  /**
   * Retrieves a job by its ID.
   * @param jobId The ID of the job to retrieve.
   * @returns The Job object or undefined if not found.
   */
  getJob(jobId: string): Job | undefined {
    return this.jobs.get(jobId);
  }

  /**
   * Updates the status and optional progress message of a job.
   * @param jobId The ID of the job to update.
   * @param status The new status for the job.
   * @param progressMessage An optional message describing the current progress.
   * @returns True if the job was found and updated, false otherwise.
   */
  updateJobStatus(jobId: string, status: JobStatus, progressMessage?: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) {
      logger.warn({ jobId }, `Attempted to update status for non-existent job.`);
      return false;
    }

    // Prevent updating status of already completed/failed jobs? Maybe allow for edge cases.
    if (job.status === JobStatus.COMPLETED || job.status === JobStatus.FAILED) {
        logger.warn({ jobId, currentStatus: job.status, newStatus: status }, `Attempted to update status of a finalized job.`);
        // Optionally return false or allow update
    }

    job.status = status;
    job.updatedAt = Date.now();
    if (progressMessage !== undefined) {
      job.progressMessage = progressMessage;
    }
    logger.info({ jobId, status, progressMessage }, `Updated job status.`);
    sseNotifier.sendProgress('auto-session', jobId, status, progressMessage);
    return true;
  }

  /**
   * Sets the final result (success or error) of a job and updates its status.
   * Automatically sets status to COMPLETED or FAILED based on result.isError.
   * @param jobId The ID of the job to set the result for.
   * @param result The final CallToolResult object.
   * @returns True if the job was found and the result was set, false otherwise.
   */
  setJobResult(jobId: string, result: CallToolResult): boolean {
    const job = this.jobs.get(jobId);
    if (!job) {
      logger.warn({ jobId }, `Attempted to set result for non-existent job.`);
      return false;
    }

    if (job.status === JobStatus.COMPLETED || job.status === JobStatus.FAILED) {
        logger.warn({ jobId, currentStatus: job.status }, `Attempted to set result for an already finalized job.`);
        // Optionally overwrite or return false
        // return false;
    }

    job.result = result;
    job.status = result.isError ? JobStatus.FAILED : JobStatus.COMPLETED;
    job.updatedAt = Date.now();
    job.progressMessage = result.isError ? 'Job failed' : 'Job completed successfully'; // Set final message

    logger.info({ jobId, finalStatus: job.status }, `Set final job result.`);
    sseNotifier.sendProgress('auto-session', jobId, job.status, job.progressMessage);
    return true;
  }

  // Optional: Add cleanup logic for old jobs if needed
  // cleanupOldJobs(maxAgeMs: number) { ... }
}

// Export a singleton instance
export const jobManager = new JobManager();
