/**
 * thinkingStripper.js
 * Strips thinking/reasoning content from model responses.
 * Supports both Gemma 4 (<|channel>thought\n...<channel|>)
 * and Qwen3.5 / DeepSeek (<think>...</think>) formats.
 */

const GEMMA4_START = '<|channel>';
const GEMMA4_END = '<channel|>';
const THOUGHT_PREFIX = 'thought\n';

const QWEN_START = '<think>';
const QWEN_END = '</think>';

/**
 * Strip thinking content from a single text string.
 * Returns { cleanText, thinking }.
 */
function stripThinking(text) {
	if (typeof text !== 'string') {
		return { cleanText: text, thinking: null };
	}

	let cleanText = text;
	let thinking = null;

	// Try Gemma 4 format first
	const gStart = cleanText.indexOf(GEMMA4_START);
	if (gStart !== -1) {
		const gEnd = cleanText.indexOf(GEMMA4_END, gStart);
		if (gEnd !== -1) {
			const fullBlock = cleanText.slice(gStart, gEnd + GEMMA4_END.length);
			let reasoning = cleanText.slice(gStart + GEMMA4_START.length, gEnd);
			// Strip "thought\n" prefix if present
			if (reasoning.startsWith(THOUGHT_PREFIX)) {
				reasoning = reasoning.slice(THOUGHT_PREFIX.length);
			}
			thinking = reasoning.trim();
			cleanText = cleanText.slice(0, gStart) + cleanText.slice(gEnd + GEMMA4_END.length);
		}
	}

	// Try Qwen / DeepSeek format
	const qStart = cleanText.indexOf(QWEN_START);
	if (qStart !== -1) {
		const qEnd = cleanText.indexOf(QWEN_END, qStart);
		if (qEnd !== -1) {
			const reasoning = cleanText.slice(qStart + QWEN_START.length, qEnd);
			if (!thinking) {
				thinking = reasoning.trim();
			} else {
				thinking += '\n' + reasoning.trim();
			}
			cleanText = cleanText.slice(0, qStart) + cleanText.slice(qEnd + QWEN_END.length);
		}
	}

	return {
		cleanText: cleanText.trim(),
		thinking: thinking,
	};
}

/**
 * Strip thinking from a LangChain AIMessage content field.
 * Content can be a string or an array of content blocks.
 */
function stripMessageContent(content) {
	if (typeof content === 'string') {
		return stripThinking(content);
	}

	if (Array.isArray(content)) {
		let allThinking = null;
		const cleanBlocks = content.map((block) => {
			if (typeof block === 'string') {
				const result = stripThinking(block);
				if (result.thinking) {
					allThinking = allThinking ? allThinking + '\n' + result.thinking : result.thinking;
				}
				return result.cleanText;
			}
			if (block && typeof block === 'object' && block.type === 'text' && typeof block.text === 'string') {
				const result = stripThinking(block.text);
				if (result.thinking) {
					allThinking = allThinking ? allThinking + '\n' + result.thinking : result.thinking;
				}
				return { ...block, text: result.cleanText };
			}
			// Non-text blocks (tool_use, etc.) pass through unchanged
			return block;
		});
		return {
			cleanText: cleanBlocks,
			thinking: allThinking,
		};
	}

	return { cleanText: content, thinking: null };
}

module.exports = {
	stripThinking,
	stripMessageContent,
};
