/**
 * AiAgentV2.node.js
 * AI Agent V2 - Text/Audio input with tool calling, memory, and thinking stripper.
 */

const { NodeConnectionTypes, NodeOperationError } = require('n8n-workflow');
const { HumanMessage } = require('@langchain/core/messages');
const { prepareAudioContent } = require('./utils/audioHandler');
const { buildMessages } = require('./utils/memoryManager');
const { runAgentLoop } = require('./utils/agentLoop');

class AiAgentV2 {
	constructor() {
		this.description = {
			displayName: 'AI Agent V2',
			name: 'aiAgentV2',
			icon: 'file:ai-agent-v2.svg',
			group: ['transform'],
			version: 1,
			subtitle: '={{$parameter["options"]["systemMessage"] ? "with system msg" : ""}}',
			description: 'AI agent with text/audio input, tool calling, and thinking support',
			defaults: {
				name: 'AI Agent V2',
			},
			inputs: [
				{
					displayName: 'Main',
					maxConnections: 1,
					type: NodeConnectionTypes.Main,
				},
				{
					displayName: 'Model',
					maxConnections: 1,
					type: NodeConnectionTypes.AiLanguageModel,
					required: true,
				},
				{
					displayName: 'Tools',
					maxConnections: undefined,
					type: NodeConnectionTypes.AiTool,
					required: false,
				},
				{
					displayName: 'Memory',
					maxConnections: 1,
					type: NodeConnectionTypes.AiMemory,
					required: false,
				},
			],
			outputs: [
				{
					displayName: 'Main',
					type: NodeConnectionTypes.Main,
				},
			],
			properties: [
				{
					displayName: 'Text',
					name: 'text',
					type: 'string',
					default: '',
					typeOptions: {
						rows: 4,
					},
					description: 'Text input to the agent (leave empty if using audio only)',
				},
				{
					displayName: 'Audio',
					name: 'audio',
					type: 'string',
					default: '',
					placeholder: 'e.g. {{ $binary.audio.data }}',
					description: 'Binary audio data property (supports .aac, .wav, .mp3)',
				},
				{
					displayName: 'Options',
					name: 'options',
					type: 'collection',
					default: {},
					placeholder: 'Add Option',
					options: [
						{
							displayName: 'System Message',
							name: 'systemMessage',
							type: 'string',
							default: 'You are a helpful assistant.',
							typeOptions: {
								rows: 6,
							},
							description: 'System message sent to the LLM',
						},
						{
							displayName: 'Max Iterations',
							name: 'maxIterations',
							type: 'number',
							default: 10,
							description: 'Maximum number of tool-calling iterations',
						},
						{
							displayName: 'Return Intermediate Steps',
							name: 'returnIntermediateSteps',
							type: 'boolean',
							default: false,
							description: 'Whether to return intermediate tool-calling steps in the output',
						},
						{
							displayName: 'Max History Messages',
							name: 'maxHistoryMessages',
							type: 'number',
							default: 20,
							description: 'Maximum number of chat history messages to load from memory',
						},
					],
				},
			],
		};
	}

	async execute() {
		const items = this.getInputData();
		const returnData = [];

		// Get connected resources
		const model = await this.getInputConnectionData(NodeConnectionTypes.AiLanguageModel, 0);
		const tools = await this.getInputConnectionData(NodeConnectionTypes.AiTool, 0) || [];
		const memory = await this.getInputConnectionData(NodeConnectionTypes.AiMemory, 0);

		// Validate model
		if (!model) {
			throw new NodeOperationError(this.getNode(), 'No language model connected. Please connect an LLM node to the "Model" input.');
		}

		// Ensure tools is an array
		const toolArray = Array.isArray(tools) ? tools : tools ? [tools] : [];

		for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
			const item = items[itemIndex];
			const text = this.getNodeParameter('text', itemIndex, '') || '';
			const audioPath = this.getNodeParameter('audio', itemIndex, '') || '';
			const options = this.getNodeParameter('options', itemIndex, {});

			const systemMessage = options.systemMessage || 'You are a helpful assistant.';
			const maxIterations = options.maxIterations || 10;
			const returnIntermediateSteps = options.returnIntermediateSteps || false;
			const maxHistoryMessages = options.maxHistoryMessages || 20;

			// Resolve audio binary data
			let audioContent = null;
			let hasAudio = false;

			if (audioPath) {
				// Parse the audio path, e.g. "audio" or "audio.data"
				const pathParts = audioPath.split('.');
				const binaryKey = pathParts[0];

				if (item.binary && item.binary[binaryKey]) {
					const binaryData = item.binary[binaryKey];
					const mimeType = binaryData.mimeType || 'audio/aac';
					const audioBuffer = Buffer.from(binaryData.data, 'base64');
					audioContent = prepareAudioContent(audioBuffer, mimeType);
					hasAudio = true;
				}
			}

			// Determine human input
			const humanInput = text.trim() || (hasAudio ? '[Audio input]' : '');

			if (!humanInput && !hasAudio) {
				throw new NodeOperationError(this.getNode(), 'No input provided. Please provide text or audio input.');
			}

			// Build message array
			const messages = await buildMessages({
				systemMessage,
				audioContent,
				humanInput: text,
				memory,
				maxHistory: maxHistoryMessages,
			});

			// Run agent loop
			const result = await runAgentLoop({
				messages,
				model,
				tools: toolArray,
				memory,
				humanInput: text,
				hasAudio,
				options: { maxIterations, systemMessage },
			});

			// Build output item
			const outputItem = {
				json: {
					output: result.output,
					hasAudio: result.hasAudio,
				},
			};

			if (returnIntermediateSteps) {
				outputItem.json.intermediateSteps = result.intermediateSteps;
			}

			if (result.thinking) {
				outputItem.json.thinking = result.thinking;
			}

			returnData.push(outputItem);
		}

		return [returnData];
	}
}

module.exports = AiAgentV2;
