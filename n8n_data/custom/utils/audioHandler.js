/**
 * audioHandler.js
 * Handles audio file conversion and preparation for LLM input.
 * Converts .aac/.mp3 → .wav via FFmpeg, then base64-encodes for LangChain.
 * Supports both in-memory and filesystem binary data modes.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

/**
 * Search for a binary data file by UUID in n8n's binaryData directory.
 */
function findBinaryFileById(uuid) {
	const baseDirs = [
		path.join(os.homedir(), '.n8n', 'binaryData'),
		path.join(os.homedir(), '.n8n', 'storage'),
		'/home/node/.n8n/binaryData',
		'/home/node/.n8n/storage',
	];

	for (const baseDir of baseDirs) {
		if (!fs.existsSync(baseDir)) continue;

		function searchDir(dir) {
			try {
				const entries = fs.readdirSync(dir, { withFileTypes: true });
				for (const entry of entries) {
					const fullPath = path.join(dir, entry.name);
					if (entry.isDirectory()) {
						const result = searchDir(fullPath);
						if (result) return result;
					} else if (entry.name === uuid) {
						return fullPath;
					}
				}
			} catch (_) {}
			return null;
		}

		const result = searchDir(baseDir);
		if (result) return result;
	}
	return null;
}

/**
 * Extract raw audio bytes from n8n binary data object.
 * Handles both in-memory (base64) and filesystem modes.
 */
function getAudioBuffer(binaryData) {
	// DEBUG: Log everything
	console.log('[AI Agent V2] === BINARY DATA DEBUG ===');
	console.log('[AI Agent V2] Keys:', Object.keys(binaryData));
	console.log('[AI Agent V2] mimeType:', binaryData.mimeType);
	console.log('[AI Agent V2] fileName:', binaryData.fileName);
	console.log('[AI Agent V2] fileSize:', binaryData.fileSize);
	console.log('[AI Agent V2] id:', binaryData.id);
	console.log('[AI Agent V2] data type:', typeof binaryData.data);
	console.log('[AI Agent V2] data length:', typeof binaryData.data === 'string' ? binaryData.data.length : (Buffer.isBuffer(binaryData.data) ? binaryData.data.length : 'N/A'));
	if (typeof binaryData.data === 'string' && binaryData.data.length < 200) {
		console.log('[AI Agent V2] data content:', binaryData.data);
	}
	
	// Case 1: data is a base64 string (in-memory mode)
	if (typeof binaryData.data === 'string' && binaryData.data.length > 100) {
		// Check if it looks like base64
		const base64Pattern = /^[A-Za-z0-9+/]*={0,2}$/;
		if (base64Pattern.test(binaryData.data.substring(0, Math.min(100, binaryData.data.length)))) {
			console.log('[AI Agent V2] Detected base64 data');
			return Buffer.from(binaryData.data, 'base64');
		}
	}

	// Case 2: data is already a Buffer
	if (Buffer.isBuffer(binaryData.data) && binaryData.data.length > 0) {
		console.log('[AI Agent V2] Detected Buffer data');
		return binaryData.data;
	}

	// Case 3: filesystem-v2 mode - id contains the path after 'filesystem-v2:'
	if (binaryData.id && binaryData.id.startsWith('filesystem-v2:')) {
		const relativePath = binaryData.id.substring('filesystem-v2:'.length);
		const baseDirs = [
			path.join(os.homedir(), '.n8n', 'storage'),
			path.join(os.homedir(), '.n8n', 'binaryData'),
			'/home/node/.n8n/storage',
			'/home/node/.n8n/binaryData',
		];
		for (const baseDir of baseDirs) {
			const fullPath = path.join(baseDir, relativePath);
			if (fs.existsSync(fullPath)) {
				console.log('[AI Agent V2] Found file via filesystem-v2 path:', fullPath);
				return fs.readFileSync(fullPath);
			}
		}
		console.log('[AI Agent V2] filesystem-v2 path not found:', relativePath);
	}

	// Case 4: filesystem v1 mode - search by id
	if (binaryData.id) {
		console.log('[AI Agent V2] Searching filesystem for id:', binaryData.id);
		const filePath = findBinaryFileById(binaryData.id);
		if (filePath) {
			console.log('[AI Agent V2] Found file:', filePath);
			return fs.readFileSync(filePath);
		}
		console.log('[AI Agent V2] File not found for id:', binaryData.id);
	}

	// Case 4: try reading from fileName
	if (binaryData.fileName && fs.existsSync(binaryData.fileName)) {
		console.log('[AI Agent V2] Reading from fileName:', binaryData.fileName);
		return fs.readFileSync(binaryData.fileName);
	}

	// Case 5: data might be a path
	if (typeof binaryData.data === 'string' && fs.existsSync(binaryData.data)) {
		console.log('[AI Agent V2] Reading from data path:', binaryData.data);
		return fs.readFileSync(binaryData.data);
	}

	// Last resort
	if (typeof binaryData.data === 'string') {
		console.log('[AI Agent V2] Last resort base64 decode');
		return Buffer.from(binaryData.data, 'base64');
	}

	throw new Error('Could not extract audio data from binary object. Available keys: ' + Object.keys(binaryData).join(', '));
}

/**
 * Convert audio buffer to wav format using FFmpeg.
 * @param {Buffer} audioBuffer - Raw audio bytes
 * @param {string} originalMimeType - e.g. 'audio/aac', 'audio/mp3', 'audio/wav'
 * @returns {Buffer} - WAV-format audio bytes
 */
function convertToWav(audioBuffer, originalMimeType) {
	// If already wav, return as-is
	if (originalMimeType === 'audio/wav' || originalMimeType === 'audio/x-wav') {
		console.log('[AI Agent V2] Already WAV, skipping conversion');
		return audioBuffer;
	}

	const tmpDir = os.tmpdir();
	const inputPath = path.join(tmpDir, `input_${Date.now()}`);
	const outputPath = path.join(tmpDir, `output_${Date.now()}.wav`);

	console.log('[AI Agent V2] Writing temp file:', inputPath, 'size:', audioBuffer.length);
	fs.writeFileSync(inputPath, audioBuffer);
	
	// Verify what was written
	const verifyStat = fs.statSync(inputPath);
	console.log('[AI Agent V2] Written file size:', verifyStat.size);
	
	// Check first few bytes
	const firstBytes = fs.readFileSync(inputPath).slice(0, 20);
	console.log('[AI Agent V2] First 20 bytes:', firstBytes.toString('hex'));

	try {
		console.log('[AI Agent V2] Running FFmpeg...');
		execSync(
			`ffmpeg -y -i "${inputPath}" -ar 16000 -ac 1 -c:a pcm_s16le "${outputPath}"`,
			{ timeout: 30000, stdio: 'pipe' }
		);
		const wavBuffer = fs.readFileSync(outputPath);
		console.log('[AI Agent V2] FFmpeg success, output size:', wavBuffer.length);
		fs.unlinkSync(inputPath);
		fs.unlinkSync(outputPath);
		return wavBuffer;
	} catch (error) {
		// Cleanup
		try { fs.unlinkSync(inputPath); } catch (_) {}
		try { fs.unlinkSync(outputPath); } catch (_) {}
		throw new Error(`FFmpeg audio conversion failed: ${error.message}`);
	}
}

/**
 * Prepare an audio HumanMessage content block for LangChain.
 * Accepts n8n binary data object and returns content blocks.
 */
function prepareAudioContent(binaryData, mimeType = 'audio/wav') {
	const audioBuffer = getAudioBuffer(binaryData);
	const wavBuffer = convertToWav(audioBuffer, mimeType);
	const base64Audio = wavBuffer.toString('base64');

	return [
		{
			type: 'audio',
			source_type: 'base64',
			mime_type: 'audio/wav',
			data: base64Audio,
		},
	];
}

module.exports = {
	getAudioBuffer,
	convertToWav,
	prepareAudioContent,
};
