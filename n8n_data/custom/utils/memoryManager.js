/**
 * memoryManager.js
 * Loads and saves chat history from/to an n8n memory node.
 * Ensures only CLEAN messages (no thinking content) are persisted.
 */

const { HumanMessage, AIMessage, SystemMessage } = require('@langchain/core/messages');
const { stripMessageContent } = require('./thinkingStripper');

/**
 * Load chat history from an n8n memory instance.
 * Returns array of BaseMessage instances.
 */
async function loadMemory(memory, maxMessages = 20) {
	if (!memory) {
		return [];
	}

	try {
		const history = await memory.chatHistory.getMessages();
		// Return last N messages
		return history.slice(-maxMessages);
	} catch (error) {
		// Memory might be empty or uninitialized
		return [];
	}
}

/**
 * Save messages to memory, stripping any thinking content first.
 * This prevents thinking from polluting future context windows.
 */
async function saveToMemory(humanInput, aiOutput, memory, intermediateSteps = []) {
	if (!memory) {
		return;
	}

	// Strip thinking from human input if it's a string
	let cleanHumanInput = humanInput;
	if (typeof humanInput === 'string') {
		const result = stripMessageContent(humanInput);
		cleanHumanInput = result.cleanText;
	}

	// Strip thinking from AI output
	let cleanAiOutput = aiOutput;
	if (typeof aiOutput === 'string') {
		const result = stripMessageContent(aiOutput);
		cleanAiOutput = result.cleanText;
	} else if (Array.isArray(aiOutput)) {
		const result = stripMessageContent(aiOutput);
		cleanAiOutput = result.cleanText;
	}

	await memory.saveContext(
		{ input: cleanHumanInput },
		{ output: cleanAiOutput }
	);
}

/**
 * Build the initial message array for the agent.
 * Includes: system message (optional), audio (if any), human input, chat history.
 */
async function buildMessages({ systemMessage, audioContent, humanInput, memory, maxHistory = 20 }) {
	const messages = [];

	// System message
	if (systemMessage) {
		messages.push(new SystemMessage(systemMessage));
	}

	// Chat history (clean)
	const history = await loadMemory(memory, maxHistory);
	if (history && history.length > 0) {
		messages.push(...history);
	}

	// Current human input (with audio if present)
	if (audioContent && audioContent.length > 0) {
		// Audio mode: content is array of blocks
		if (typeof humanInput === 'string' && humanInput.trim()) {
			messages.push(new HumanMessage([
				{ type: 'text', text: humanInput },
				...audioContent,
			]));
		} else {
			messages.push(new HumanMessage(audioContent));
		}
	} else {
		// Text mode
		messages.push(new HumanMessage(humanInput));
	}

	return messages;
}

module.exports = {
	loadMemory,
	saveToMemory,
	buildMessages,
};
