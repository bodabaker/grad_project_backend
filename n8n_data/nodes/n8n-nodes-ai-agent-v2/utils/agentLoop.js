/**
 * agentLoop.js
 * Custom agent execution loop with thinking stripping and audio support.
 * Does NOT use LangChain's AgentExecutor — full manual control.
 */

const { AIMessage, ToolMessage } = require('@langchain/core/messages');
const { stripMessageContent } = require('./thinkingStripper');
const { saveToMemory } = require('./memoryManager');

/**
 * Execute a single tool call.
 * @param {Object} toolCall - { id, name, args }
 * @param {Array} tools - Array of LangChain tool instances
 * @returns {Promise<{toolCallId, toolName, toolInput, observation, error}>}
 */
async function executeTool(toolCall, tools) {
	const toolName = toolCall.name || toolCall.function?.name;
	const toolCallId = toolCall.id || toolCall.function?.id || `call_${Date.now()}`;
	let toolInput = toolCall.args || toolCall.function?.arguments || {};

	// Parse args if it's a string
	if (typeof toolInput === 'string') {
		try {
			toolInput = JSON.parse(toolInput);
		} catch {
			// Keep as string if parse fails
		}
	}

	const tool = tools.find((t) => t.name === toolName);
	if (!tool) {
		return {
			toolCallId,
			toolName,
			toolInput,
			observation: `Error: Tool "${toolName}" not found.`,
			error: true,
		};
	}

	try {
		const observation = await tool.invoke(toolInput);
		return {
			toolCallId,
			toolName,
			toolInput,
			observation: typeof observation === 'string' ? observation : JSON.stringify(observation),
			error: false,
		};
	} catch (error) {
		return {
			toolCallId,
			toolName,
			toolInput,
			observation: `Error: ${error.message}`,
			error: true,
		};
	}
}

/**
 * Run the custom agent loop.
 * @param {Object} params
 * @param {Array} params.messages - Initial message array (system + history + human input)
 * @param {Object} params.model - LangChain chat model instance
 * @param {Array} params.tools - Array of LangChain tool instances
 * @param {Object} params.memory - Optional n8n memory instance
 * @param {string} params.humanInput - Original human input (for memory save)
 * @param {boolean} params.hasAudio - Whether input included audio
 * @param {Object} params.options - { maxIterations, systemMessage }
 * @returns {Promise<{output: string, hasAudio: boolean, intermediateSteps: Array}>}
 */
async function runAgentLoop({ messages, model, tools, memory, humanInput, hasAudio, options = {} }) {
	const maxIterations = options.maxIterations || 15;
	const intermediateSteps = [];

	let currentMessages = [...messages];
	let finalOutput = '';
	let finalThinking = null;

	for (let iteration = 0; iteration < maxIterations; iteration++) {
		// Call the LLM
		const response = await model.invoke(currentMessages);

		// Strip thinking from response content
		const stripResult = stripMessageContent(response.content);
		const cleanContent = stripResult.cleanText;
		const thinking = stripResult.thinking;

		// Check for tool calls in the response
		const toolCalls = response.tool_calls || response.additional_kwargs?.tool_calls || [];

		if (toolCalls && toolCalls.length > 0) {
			// --- TOOL CALLING ITERATION ---

			// Add CLEAN AIMessage (with tool_calls but no thinking) to messages
			const cleanAiMessage = new AIMessage({
				content: cleanContent,
				tool_calls: toolCalls,
				additional_kwargs: {
					...response.additional_kwargs,
					// Remove any reasoning content from kwargs
				},
			});
			currentMessages.push(cleanAiMessage);

			// Execute each tool call
			const stepResults = [];
			for (const toolCall of toolCalls) {
				const result = await executeTool(toolCall, tools);
				stepResults.push(result);

				// Add ToolMessage to conversation
				currentMessages.push(
					new ToolMessage({
						content: result.observation,
						tool_call_id: result.toolCallId,
						name: result.toolName,
					})
				);
			}

			// Record intermediate step
			intermediateSteps.push({
				tool: stepResults[0]?.toolName || 'unknown',
				toolInput: stepResults[0]?.toolInput || {},
				observation: stepResults.map((r) => r.observation).join('\n'),
				thinking: thinking,
			});
		} else {
			// --- FINAL ANSWER ---
			finalOutput = typeof cleanContent === 'string' ? cleanContent : JSON.stringify(cleanContent);
			finalThinking = thinking;

			// Add clean AIMessage to messages for memory
			currentMessages.push(
				new AIMessage({
					content: cleanContent,
				})
			);

			break;
		}
	}

	// Save clean messages to memory
	if (memory) {
		try {
			await saveToMemory(humanInput, finalOutput, memory, intermediateSteps);
		} catch (error) {
			// Memory save error shouldn't fail the whole execution
			console.warn('Memory save failed:', error.message);
		}
	}

	return {
		output: finalOutput,
		hasAudio,
		intermediateSteps,
		thinking: finalThinking,
	};
}

module.exports = {
	runAgentLoop,
	executeTool,
};
