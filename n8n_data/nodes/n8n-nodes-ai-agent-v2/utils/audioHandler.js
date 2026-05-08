/**
 * audioHandler.js
 * Handles audio file conversion and preparation for LLM input.
 * Converts .aac/.mp3 → .wav via FFmpeg, then base64-encodes for LangChain.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

/**
 * Convert audio buffer to wav format using FFmpeg.
 * @param {Buffer} audioBuffer - Raw audio bytes
 * @param {string} originalMimeType - e.g. 'audio/aac', 'audio/mp3', 'audio/wav'
 * @returns {Buffer} - WAV-format audio bytes
 */
function convertToWav(audioBuffer, originalMimeType) {
	// If already wav, return as-is
	if (originalMimeType === 'audio/wav' || originalMimeType === 'audio/x-wav') {
		return audioBuffer;
	}

	const tmpDir = os.tmpdir();
	const inputPath = path.join(tmpDir, `input_${Date.now()}`);
	const outputPath = path.join(tmpDir, `output_${Date.now()}.wav`);

	fs.writeFileSync(inputPath, audioBuffer);

	try {
		execSync(
			`ffmpeg -y -i "${inputPath}" -ar 16000 -ac 1 -c:a pcm_s16le "${outputPath}"`,
			{ timeout: 30000, stdio: 'pipe' }
		);
		const wavBuffer = fs.readFileSync(outputPath);
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
 * Returns a content block array suitable for LangChain's HumanMessage.
 */
function prepareAudioContent(audioBuffer, mimeType = 'audio/wav') {
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
	convertToWav,
	prepareAudioContent,
};
