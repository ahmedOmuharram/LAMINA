/**
 * Utility functions for parsing and handling tool call outputs
 */

/**
 * Parse tool output and check for standardized result wrapper format
 */
export function parseToolOutput(output: any) {
  // If no output, return null status
  if (!output) {
    return { 
      parsed: null, 
      hasWrapper: false, 
      success: null,
      error: null,
      errorType: null
    };
  }

  // Try to parse if it's a string
  let parsedOutput = output;
  if (typeof output === 'string') {
    try {
      parsedOutput = JSON.parse(output);
    } catch {
      // If parsing fails, assume no wrapper
      return { 
        parsed: output, 
        hasWrapper: false, 
        success: null,
        error: null,
        errorType: null
      };
    }
  }
  
  // Check if output follows the standardized result wrapper format
  const hasWrapper = parsedOutput && typeof parsedOutput === 'object' && 'success' in parsedOutput;
  
  return {
    parsed: parsedOutput,
    hasWrapper,
    success: hasWrapper ? parsedOutput.success : null,
    error: hasWrapper ? parsedOutput.error : null,
    errorType: hasWrapper ? parsedOutput.error_type : null,
    data: hasWrapper ? parsedOutput.data : null,
    metadata: hasWrapper ? parsedOutput.metadata : null,
    warnings: hasWrapper ? parsedOutput.warnings : null,
    suggestions: hasWrapper ? parsedOutput.suggestions : null,
  };
}

/**
 * Get the actual status of a tool call based on its output
 * Checks the standardized result wrapper's success field
 */
export function getToolActualStatus(
  streamingStatus: 'started' | 'completed' | 'error',
  output?: any
): 'started' | 'completed' | 'error' {
  // If still running, keep 'started' status
  if (streamingStatus === 'started') {
    return 'started';
  }
  
  // Parse output to check success field
  const outputInfo = parseToolOutput(output);
  
  // If we have a standardized wrapper with success field, use it
  if (outputInfo.hasWrapper && outputInfo.success !== null) {
    return outputInfo.success ? 'completed' : 'error';
  }
  
  // Fall back to streaming status
  return streamingStatus;
}

/**
 * Get a display-friendly status label
 */
export function getToolStatusLabel(status: 'started' | 'completed' | 'error'): string {
  switch (status) {
    case 'started':
      return 'running';
    case 'completed':
      return 'success';
    case 'error':
      return 'failed';
    default:
      return status;
  }
}

