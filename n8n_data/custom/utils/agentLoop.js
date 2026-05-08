/**
 * agentLoop.js
 * Custom agent execution loop with thinking stripping, audio support, and streaming.
 * Matches built-in AI Agent node's streaming behavior.
 */

const { AIMessage, ToolMessage } = require('@langchain/core/messages');
const { stripMessageContent } = require('./thinkingStripper');
const { saveToMemory } = require('./memoryManager');

/**
 * Execute a single tool call.
 */
async function executeTool(toolCall, tools) {
	const toolName = toolCall.name || toolCall.function?.name;
	const toolCallId = toolCall.id || toolCall.function?.id || `call_${Date.now()}`;
	let toolInput = toolCall.args || toolCall.function?.arguments || {};

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
 * Parse tool calls from model response.
 */
function parseToolCalls(response) {
	let toolCalls = response.tool_calls || response.additional_kwargs?.tool_calls || [];
	if (toolCalls && toolCalls.length > 0) {
		return toolCalls;
	}

	// Text-based fallback
	const content = typeof response.content === 'string' ? response.content : '';

	const toolCallMatch = content.match(/<\|tool_call\>call:\s*(\w+)\s*(.*?)\n/i);
	if (toolCallMatch) {
		return [{
			id: `call_${Date.now()}`,
			name: toolCallMatch[1],
			args: toolCallMatch[2].trim(),
		}];
	}

	const xmlMatch = content.match(/<(\w+)>(.*?)<\/\1>/s);
	if (xmlMatch) {
		return [{
			id: `call_${Date.now()}`,
			name: xmlMatch[1],
			args: xmlMatch[2].trim(),
		}];
	}

	return [];
}

/**
 * Process event stream from model.streamEvents(), exactly like built-in agent.
 */
async function processEventStream(ctx, eventStream, itemIndex) {
	const agentResult = {
		output: '',
	};
	const toolCalls = [];

	ctx.sendChunk('begin', itemIndex);

	for await (const event of eventStream) {
		switch (event.event) {
			case 'on_chat_model_stream': {
				const chunk = event.data?.chunk;
				if (chunk?.content) {
					const chunkContent = chunk.content;
					let chunkText = '';
					if (Array.isArray(chunkContent)) {
						for (const message of chunkContent) {
							if (message?.type === 'text') {
								chunkText += message?.text;
							}
						}
					} else if (typeof chunkContent === 'string') {
						chunkText = chunkContent;
					}
					ctx.sendChunk('item', itemIndex, chunkText);
					agentResult.output += chunkText;
				}
				break;
			}
			case 'on_chat_model_end': {
				if (event.data) {
					const chatModelData = event.data;
					const output = chatModelData.output;
					if (output?.tool_calls && output.tool_calls.length > 0) {
						for (const toolCall of output.tool_calls) {
							toolCalls.push({
								tool: toolCall.name,
								toolInput: toolCall.args,
								toolCallId: toolCall.id || 'unknown',
								type: toolCall.type || 'tool_call',
								log: output.content || `Calling ${toolCall.name} with input: ${JSON.stringify(toolCall.args)}`,
								messageLog: [output],
								additionalKwargs: output.additional_kwargs,
							});
						}
					}
				}
				break;
			}
			default:
				break;
		}
	}

	ctx.sendChunk('end', itemIndex);

	if (toolCalls.length > 0) {
		agentResult.toolCalls = toolCalls;
	}

	return agentResult;
}

/**
 * Run the custom agent loop.
 */
async function runAgentLoop({ messages, model, tools, memory, humanInput, hasAudio, options = {}, ctx, itemIndex }) {
	const maxIterations = options.maxIterations || 15;
	const enableStreaming = options.enableStreaming || false;
	const intermediateSteps = [];

	let currentMessages = [...messages];
	let finalOutput = '';
	let finalThinking = null;

	// Bind tools to model
	let modelWithTools = model;
	if (tools && tools.length > 0 && typeof model.bindTools === 'function') {
		try {
			modelWithTools = model.bindTools(tools);
		} catch (error) {
			console.warn('[AI Agent V2] Failed to bind tools to model:', error.message);
		}
	}

	// Check if streaming is available (same check as built-in agent)
	const isStreamingAvailable = 'isStreaming' in ctx ? ctx.isStreaming?.() : false;
	const useStreaming = enableStreaming && isStreamingAvailable;

	for (let iteration = 0; iteration < maxIterations; iteration++) {
		console.log(`[AI Agent V2] Iteration ${iteration + 1}/${maxIterations}, streaming=${useStreaming}`);

		let response;

		if (useStreaming) {
			// --- STREAMING MODE (matches built-in agent exactly) ---
			const eventStream = modelWithTools.streamEvents(
				currentMessages,
				{ version: 'v2' }
			);

			const streamResult = await processEventStream(ctx, eventStream, itemIndex);

			if (streamResult.toolCalls && streamResult.toolCalls.length > 0) {
				// Tool calls detected during stream - execute them
				const toolCallsList = streamResult.toolCalls.map(tc => ({
					id: tc.toolCallId,
					name: tc.tool,
					args: tc.toolInput,
				}));

				// Add AIMessage with tool calls
				currentMessages.push(
					new AIMessage({
						content: streamResult.output || '',
						tool_calls: toolCallsList,
					})
				);

				// Execute tools
				const stepResults = [];
				for (const toolCall of toolCallsList) {
					const result = await executeTool(toolCall, tools);
					stepResults.push(result);

					currentMessages.push(
						new ToolMessage({
							content: result.observation,
							tool_call_id: result.toolCallId,
							name: result.toolName,
						})
					);
				}

				intermediateSteps.push({
					tool: stepResults[0]?.toolName || 'unknown',
					toolInput: stepResults[0]?.toolInput || {},
					observation: stepResults.map((r) => r.observation).join('\n'),
					thinking: null,
				});

				continue; // Next iteration
			} else {
				// No tool calls - this is the final answer
				finalOutput = streamResult.output;
				finalThinking = null;

				currentMessages.push(
					new AIMessage({
						content: finalOutput,
					})
				);

				break;
			}
		} else {
			// --- NON-STREAMING MODE ---
			response = await modelWithTools.invoke(currentMessages);

			const stripResult = stripMessageContent(response.content);
			const cleanContent = stripResult.cleanText;
			const thinking = stripResult.thinking;

			const toolCalls = parseToolCalls(response);

			if (toolCalls && toolCalls.length > 0) {
				// Tool calling iteration
				const cleanAiMessage = new AIMessage({
					content: cleanContent,
					tool_calls: response.tool_calls || undefined,
					additional_kwargs: {
						...response.additional_kwargs,
					},
				});
				currentMessages.push(cleanAiMessage);

				const stepResults = [];
				for (const toolCall of toolCalls) {
					const result = await executeTool(toolCall, tools);
					stepResults.push(result);

					currentMessages.push(
						new ToolMessage({
							content: result.observation,
							tool_call_id: result.toolCallId,
							name: result.toolName,
						})
					);
				}

				intermediateSteps.push({
					tool: stepResults[0]?.toolName || 'unknown',
					toolInput: stepResults[0]?.toolInput || {},
					observation: stepResults.map((r) => r.observation).join('\n'),
					thinking: thinking,
				});
			} else {
				// Final answer
				finalOutput = typeof cleanContent === 'string' ? cleanContent : JSON.stringify(cleanContent);
				finalThinking = thinking;

				currentMessages.push(
					new AIMessage({
						content: cleanContent,
					})
				);

				break;
			}
		}
	}

	// Save clean messages to memory
	if (memory) {
		try {
			await saveToMemory(humanInput, finalOutput, memory, intermediateSteps);
		} catch (error) {
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
