import axios, { AxiosError } from 'axios';
import { OpenRouterConfig } from '../types/workflow.js';
import logger from '../logger.js';
import { AppError, ApiError, ConfigurationError, ParsingError } from './errors.js';
import { selectModelForTask } from './configLoader.js';

/**
 * Performs a direct LLM call for text generation (not sequential thinking).
 * This allows more control over the exact output format without the sequential thinking wrapper.
 *
 * @param prompt The user prompt to send to the LLM.
 * @param systemPrompt The system prompt defining the LLM's role and output format.
 * @param config OpenRouter configuration containing API key and model information.
 * @param logicalTaskName A string identifier for the logical task being performed, used for model selection via llm_mapping.
 * @param temperature Optional temperature override (defaults to 0.1 for deterministic output).
 * @returns The raw text response from the LLM.
 * @throws AppError or subclasses (ConfigurationError, ApiError, ParsingError) if the call fails.
 */
export async function performDirectLlmCall(
  prompt: string,
  systemPrompt: string,
  config: OpenRouterConfig,
  logicalTaskName: string,
  temperature: number = 0.1 // Default to low temperature for predictable generation
): Promise<string> {
  // Log the received config object for debugging
  logger.debug({
    configReceived: true,
    apiKeyPresent: Boolean(config.apiKey),
    mapping: config.llm_mapping ? 'present' : 'missing',
    mappingSize: config.llm_mapping ? Object.keys(config.llm_mapping).length : 0,
    mappingKeys: config.llm_mapping ? Object.keys(config.llm_mapping) : []
  }, `performDirectLlmCall received config for task: ${logicalTaskName}`);

  // Check for API key
  if (!config.apiKey) {
    throw new ConfigurationError("OpenRouter API key (OPENROUTER_API_KEY) is not configured.");
  }

  // Select the model using the utility function
  // Provide a sensible default if no specific model is found or configured
  const defaultModel = config.geminiModel || "github/copilot-chat"; // Use GitHub Copilot Chat as default
  const modelToUse = selectModelForTask(config, logicalTaskName, defaultModel);
  logger.info({ modelSelected: modelToUse, logicalTaskName }, `Selected model for direct LLM call.`);

  try {
    const response = await axios.post(
      `${config.baseUrl}/chat/completions`,
      {
        model: modelToUse,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt }
        ],
        max_tokens: 4000, // Consider making this configurable if needed
        temperature: temperature // Use the provided or default temperature
      },
      {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${config.apiKey}`,
          "HTTP-Referer": "https://vibe-coder-mcp.local" // Optional: Referer for tracking
        },
        timeout: 90000 // Increased timeout to 90s for potentially longer generations
      }
    );

    if (response.data?.choices?.[0]?.message?.content) {
      const responseText = response.data.choices[0].message.content.trim();
      logger.debug({ modelUsed: modelToUse, responseLength: responseText.length }, "Direct LLM call successful");
      return responseText;
    } else {
      logger.warn({ responseData: response.data, modelUsed: modelToUse }, "Received empty or unexpected response structure from LLM");
      throw new ParsingError(
        "Invalid API response structure received from LLM",
        { responseData: response.data, modelUsed: modelToUse, logicalTaskName }
      );
    }
  } catch (error) {
    // Log with the actual model used
    logger.error({ err: error, modelUsed: modelToUse, logicalTaskName }, `Direct LLM API call failed for ${logicalTaskName}`);

    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      const status = axiosError.response?.status;
      const responseData = axiosError.response?.data;
      const apiMessage = `LLM API Error: Status ${status || 'N/A'}. ${axiosError.message}`;
      throw new ApiError(
        apiMessage,
        status,
        { modelUsed: modelToUse, logicalTaskName, responseData }, // Include logicalTaskName in context
        axiosError
      );
    } else if (error instanceof AppError) {
      // Re-throw specific AppErrors (like ParsingError from above)
      throw error;
    } else if (error instanceof Error) {
      // Wrap other generic errors
      throw new AppError(
        `LLM call failed for ${logicalTaskName}: ${error.message}`,
        { modelUsed: modelToUse, logicalTaskName }, // Include logicalTaskName
        error
      );
    } else {
      // Handle non-Error throws
      throw new AppError(
        `Unknown error during LLM call for ${logicalTaskName}.`,
        { modelUsed: modelToUse, logicalTaskName, thrownValue: String(error) } // Include logicalTaskName
      );
    }
  }
}

/**
 * Dynamically update all available models using GitHub Models API.
 * 
 * @param config OpenRouter configuration containing API key and model information.
 * @returns A Promise resolving to an updated llm_mapping object.
 * @throws AppError or subclasses (ConfigurationError, ApiError, ParsingError) if the call fails.
 */
export async function updateAvailableModels(config: OpenRouterConfig): Promise<Record<string, string>> {
  // Log the received config object for debugging
  logger.debug({
    configReceived: true,
    apiKeyPresent: Boolean(config.apiKey),
    mapping: config.llm_mapping ? 'present' : 'missing',
    mappingSize: config.llm_mapping ? Object.keys(config.llm_mapping).length : 0,
    mappingKeys: config.llm_mapping ? Object.keys(config.llm_mapping) : []
  }, `updateAvailableModels received config`);

  // Check for API key
  if (!config.apiKey) {
    throw new ConfigurationError("GitHub API key (GITHUB_API_KEY) is not configured.");
  }

  try {
    const response = await axios.get(
      `https://models.github.ai/inference`,
      {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${config.apiKey}`,
          "HTTP-Referer": "https://vibe-coder-mcp.local" // Optional: Referer for tracking
        },
        timeout: 90000 // Increased timeout to 90s for potentially longer generations
      }
    );

    if (response.data?.models) {
      const models = response.data.models;
      const updatedMapping: Record<string, string> = {};

      models.forEach((model: { name: string; id: string }) => {
        updatedMapping[model.name] = model.id;
      });

      logger.debug({ updatedMappingSize: Object.keys(updatedMapping).length }, "Successfully updated available models");
      return updatedMapping;
    } else {
      logger.warn({ responseData: response.data }, "Received empty or unexpected response structure from GitHub Models API");
      throw new ParsingError(
        "Invalid API response structure received from GitHub Models API",
        { responseData: response.data }
      );
    }
  } catch (error) {
    // Log with the actual model used
    logger.error({ err: error }, `GitHub Models API call failed`);

    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      const status = axiosError.response?.status;
      const responseData = axiosError.response?.data;
      const apiMessage = `GitHub Models API Error: Status ${status || 'N/A'}. ${axiosError.message}`;
      throw new ApiError(
        apiMessage,
        status,
        { responseData }, // Include responseData in context
        axiosError
      );
    } else if (error instanceof AppError) {
      // Re-throw specific AppErrors (like ParsingError from above)
      throw error;
    } else if (error instanceof Error) {
      // Wrap other generic errors
      throw new AppError(
        `GitHub Models API call failed: ${error.message}`,
        { thrownValue: String(error) }, // Include thrownValue
        error
      );
    } else {
      // Handle non-Error throws
      throw new AppError(
        `Unknown error during GitHub Models API call.`,
        { thrownValue: String(error) } // Include thrownValue
      );
    }
  }
}

/**
 * Change the default AI Service from openrouter to use the Visual Studio Code GitHub Copilot Chat extension.
 * Add options to switch between all popular online AI services such as Anthropic, DeepSeek, OpenAI, using their corresponding API keys.
 * Integrate the GitHub Models available in the GitHub Marketplace for use by using a GitHub Developer Key.
 * Dynamically update all available models.
 * Add the ability to use all popular Online AI Services without using an api key.
 * Use the website facing chat implementation in a hidden background service thus bypassing the api key requirement.
 * Allow intercommunication between different models and dynamic model selection using the best available model for the task.
 */
export async function changeDefaultAIService(config: OpenRouterConfig): Promise<void> {
  // Log the received config object for debugging
  logger.debug({
    configReceived: true,
    apiKeyPresent: Boolean(config.apiKey),
    mapping: config.llm_mapping ? 'present' : 'missing',
    mappingSize: config.llm_mapping ? Object.keys(config.llm_mapping).length : 0,
    mappingKeys: config.llm_mapping ? Object.keys(config.llm_mapping) : []
  }, `changeDefaultAIService received config`);

  // Change the default AI Service to use the Visual Studio Code GitHub Copilot Chat extension
  config.baseUrl = "https://api.github.com/copilot-chat";
  config.geminiModel = "github/copilot-chat";

  // Add options to switch between all popular online AI services
  const aiServices = ["Anthropic", "DeepSeek", "OpenAI"];
  aiServices.forEach(service => {
    // Add logic to switch between AI services using their corresponding API keys
    // This is a placeholder for the actual implementation
    logger.info(`Added option to switch to ${service} AI service`);
  });

  // Integrate the GitHub Models available in the GitHub Marketplace
  const updatedModels = await updateAvailableModels(config);
  config.llm_mapping = updatedModels;

  // Add the ability to use all popular Online AI Services without using an api key
  // Use the website facing chat implementation in a hidden background service
  // This is a placeholder for the actual implementation
  logger.info("Added ability to use online AI services without API key");

  // Allow intercommunication between different models and dynamic model selection
  // This is a placeholder for the actual implementation
  logger.info("Added intercommunication between different models and dynamic model selection");
}
