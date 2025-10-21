import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { apiClient } from '@/lib/api';
import { ToolCallModal } from '@/components/chat/ToolCallModal';
import type { TestRun, TestTemplate, ToolCall } from '@/types/api';

interface TestingInterfaceProps {
  selectedModel: string;
}

export function TestingInterface({ selectedModel }: TestingInterfaceProps) {
  const [view, setView] = useState<'create' | 'results' | 'history' | 'templates'>('create');
  const [testName, setTestName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [questions, setQuestions] = useState('');
  const [currentTestRun, setCurrentTestRun] = useState<TestRun | null>(null);
  const [testRuns, setTestRuns] = useState<TestRun[]>([]);
  const [templates, setTemplates] = useState<TestTemplate[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [runInParallel, setRunInParallel] = useState(false);
  const [runningQuestions, setRunningQuestions] = useState<Set<number>>(new Set());
  const [selectedToolCall, setSelectedToolCall] = useState<ToolCall | null>(null);
  const [isToolModalOpen, setIsToolModalOpen] = useState(false);
  const [filterText, setFilterText] = useState('');
  const [filterRegex, setFilterRegex] = useState<RegExp | null>(null);
  const [filterError, setFilterError] = useState<string>('');
  const [extractionPattern, setExtractionPattern] = useState('');
  const [extractionColumns, setExtractionColumns] = useState('');
  const [extractionError, setExtractionError] = useState('');
  const [extractedData, setExtractedData] = useState<Array<Record<string, string>>>([]);
  const [showExtraction, setShowExtraction] = useState(false);
  const [extractUniqueOnly, setExtractUniqueOnly] = useState(false);

  // Load test runs from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('test_runs');
    if (saved) {
      try {
        setTestRuns(JSON.parse(saved));
      } catch (error) {
        console.error('Failed to load test runs:', error);
      }
    }
  }, []);

  // Load templates from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('test_templates');
    if (saved) {
      try {
        setTemplates(JSON.parse(saved));
      } catch (error) {
        console.error('Failed to load templates:', error);
      }
    }
  }, []);

  // Save test runs to localStorage
  const saveTestRuns = (runs: TestRun[]) => {
    setTestRuns(runs);
    localStorage.setItem('test_runs', JSON.stringify(runs));
  };

  // Save templates to localStorage
  const saveTemplates = (temps: TestTemplate[]) => {
    setTemplates(temps);
    localStorage.setItem('test_templates', JSON.stringify(temps));
  };

  // Handle filter regex update
  useEffect(() => {
    if (!filterText.trim()) {
      setFilterRegex(null);
      setFilterError('');
      return;
    }

    try {
      const regex = new RegExp(filterText, 'i');
      setFilterRegex(regex);
      setFilterError('');
    } catch (error) {
      setFilterError(error instanceof Error ? error.message : 'Invalid regex');
      setFilterRegex(null);
    }
  }, [filterText]);

  // Create a new test run
  const handleCreateTest = () => {
    if (!testName.trim() || !prompt.trim() || !questions.trim()) {
      alert('Please fill in all fields');
      return;
    }

    const questionList = questions.split('\n').filter(q => q.trim());
    if (questionList.length === 0) {
      alert('Please add at least one question');
      return;
    }

    const newTestRun: TestRun = {
      id: `test-${Date.now()}`,
      name: testName.trim(),
      prompt: prompt.trim(),
      questions: questionList.map((q, idx) => ({
        id: `q-${idx}`,
        question: q.trim(),
      })),
      model: selectedModel,
      createdAt: new Date().toISOString(),
      status: 'draft',
    };

    setCurrentTestRun(newTestRun);
    setView('results');
  };

  // Run a single question
  const runSingleQuestion = async (
    question: TestRun['questions'][0],
    testRun: TestRun
  ) => {
    const fullMessage = `${testRun.prompt}\n\n${question.question}`;
    const startTime = Date.now();

    try {
      let answer = '';
      const toolCalls: ToolCall[] = [];
      const toolCallsMap = new Map<string, ToolCall>();
      
      for await (const chunk of apiClient.streamMessage({
        messages: [{
          role: 'user' as const,
          content: fullMessage
        }],
        model: testRun.model,
        stream: true,
      })) {
        try {
          const parsed = JSON.parse(chunk);
          const delta = parsed.choices?.[0]?.delta;
          
          if (delta?.content) {
            answer += delta.content;
          }

          // Track tool calls
          if (delta?.tool_call) {
            const toolCall = delta.tool_call;
            const existingCall = toolCallsMap.get(toolCall.id);
            
            if (toolCall.status === 'started') {
              const newToolCall: ToolCall = {
                id: toolCall.id,
                name: toolCall.name,
                timestamp: Date.now(),
                duration: 0,
                status: 'started',
                input: toolCall.input,
              };
              toolCallsMap.set(toolCall.id, newToolCall);
              toolCalls.push(newToolCall);
            } else if (toolCall.status === 'completed' && existingCall) {
              existingCall.status = 'completed';
              existingCall.duration = toolCall.duration || 0;
              existingCall.output = toolCall.output;
            }
          }
        } catch (e) {
          // Ignore parse errors
        }
      }

      const duration = (Date.now() - startTime) / 1000;

      return {
        ...question,
        answer,
        timestamp: new Date().toISOString(),
        duration,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      };
    } catch (error) {
      return {
        ...question,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      };
    }
  };

  // Run the test (sequential or parallel)
  const handleRunTest = async () => {
    if (!currentTestRun) return;

    setIsRunning(true);
    setCurrentQuestionIndex(0);
    setRunningQuestions(new Set());

    const updatedTestRun = { ...currentTestRun, status: 'running' as const };
    setCurrentTestRun(updatedTestRun);

    try {
      if (runInParallel) {
        // Parallel execution
        const running = new Set(currentTestRun.questions.map((_, i) => i));
        setRunningQuestions(running);

        const results = await Promise.all(
          currentTestRun.questions.map((question) =>
            runSingleQuestion(question, updatedTestRun)
          )
        );

        updatedTestRun.questions = results;
        setCurrentTestRun({ ...updatedTestRun });
      } else {
        // Sequential execution
        for (let i = 0; i < currentTestRun.questions.length; i++) {
          setCurrentQuestionIndex(i);
          setRunningQuestions(new Set([i]));

          const result = await runSingleQuestion(
            currentTestRun.questions[i],
            updatedTestRun
          );

          updatedTestRun.questions[i] = result;
          setCurrentTestRun({ ...updatedTestRun });
        }
      }

      const completedTestRun = { ...updatedTestRun, status: 'completed' as const, completedAt: new Date().toISOString() };
      setCurrentTestRun(completedTestRun);

      // Save to test runs
      const updatedRuns = [...testRuns, completedTestRun];
      saveTestRuns(updatedRuns);
    } catch (error) {
      console.error('Test run error:', error);
      const errorTestRun = { ...updatedTestRun, status: 'error' as const };
      setCurrentTestRun(errorTestRun);
    } finally {
      setIsRunning(false);
      setRunningQuestions(new Set());
    }
  };

  // Update notes for a question
  const handleUpdateNotes = (questionId: string, notes: string) => {
    if (!currentTestRun) return;

    const updatedQuestions = currentTestRun.questions.map(q =>
      q.id === questionId ? { ...q, notes } : q
    );

    const updatedTestRun = { ...currentTestRun, questions: updatedQuestions };
    setCurrentTestRun(updatedTestRun);

    // Update in saved test runs if it exists
    const runIndex = testRuns.findIndex(r => r.id === currentTestRun.id);
    if (runIndex !== -1) {
      const updatedRuns = [...testRuns];
      updatedRuns[runIndex] = updatedTestRun;
      saveTestRuns(updatedRuns);
    }
  };

  // Export to CSV
  const handleExportCSV = (testRun: TestRun) => {
    const headers = ['Question', 'Answer', 'Notes', 'Duration (s)', 'Timestamp', 'Error'];
    const rows = testRun.questions.map(q => [
      q.question,
      q.answer || '',
      q.notes || '',
      q.duration?.toFixed(2) || '',
      q.timestamp || '',
      q.error || '',
    ]);

    const csvContent = [
      `Test Run: ${testRun.name}`,
      `Prompt: ${testRun.prompt}`,
      `Model: ${testRun.model}`,
      `Created: ${new Date(testRun.createdAt).toLocaleString()}`,
      '',
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${testRun.name.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // View an existing test run
  const handleViewTestRun = (testRun: TestRun) => {
    setCurrentTestRun(testRun);
    setView('results');
  };

  // Delete a test run
  const handleDeleteTestRun = (testRunId: string) => {
    if (confirm('Are you sure you want to delete this test run?')) {
      const updatedRuns = testRuns.filter(r => r.id !== testRunId);
      saveTestRuns(updatedRuns);
      if (currentTestRun?.id === testRunId) {
        setCurrentTestRun(null);
        setView('create');
      }
    }
  };

  // Save current form as template
  const handleSaveAsTemplate = () => {
    if (!testName.trim() || !prompt.trim() || !questions.trim()) {
      alert('Please fill in all fields before saving as template');
      return;
    }

    const questionList = questions.split('\n').filter(q => q.trim());
    if (questionList.length === 0) {
      alert('Please add at least one question');
      return;
    }

    const newTemplate: TestTemplate = {
      id: `template-${Date.now()}`,
      name: testName.trim(),
      prompt: prompt.trim(),
      questions: questionList,
      createdAt: new Date().toISOString(),
    };

    const updatedTemplates = [...templates, newTemplate];
    saveTemplates(updatedTemplates);
    alert('Template saved successfully!');
  };

  // Load a template into the form
  const handleLoadTemplate = (template: TestTemplate) => {
    setTestName(template.name);
    setPrompt(template.prompt);
    setQuestions(template.questions.join('\n'));
    setView('create');
  };

  // Delete a template
  const handleDeleteTemplate = (templateId: string) => {
    if (confirm('Are you sure you want to delete this template?')) {
      const updatedTemplates = templates.filter(t => t.id !== templateId);
      saveTemplates(updatedTemplates);
    }
  };

  // Open tool call modal
  const handleToolClick = (tool: ToolCall) => {
    setSelectedToolCall(tool);
    setIsToolModalOpen(true);
  };

  // Filter function for questions
  const matchesFilter = (question: TestRun['questions'][0]): boolean => {
    if (!filterRegex) return true;

    const searchText = [
      question.question,
      question.answer || '',
      question.notes || '',
      question.error || '',
    ].join(' ');

    return filterRegex.test(searchText);
  };

  // Extract data from test results
  const handleExtractData = () => {
    if (!currentTestRun) return;
    if (!extractionPattern.trim()) {
      setExtractionError('Please enter a regex pattern');
      return;
    }

    try {
      // Parse column names
      const columnNames = extractionColumns
        .split(',')
        .map(c => c.trim())
        .filter(c => c);

      if (columnNames.length === 0) {
        setExtractionError('Please enter at least one column name');
        return;
      }

      // Create regex with global flag
      const regex = new RegExp(extractionPattern, 'gi');
      const extracted: Array<Record<string, string>> = [];

      // Extract from all filtered questions
      currentTestRun.questions.filter(matchesFilter).forEach((question) => {
        const searchText = [
          question.question,
          question.answer || '',
          question.notes || '',
        ].join('\n');

        // Find all matches
        let match;
        while ((match = regex.exec(searchText)) !== null) {
          const row: Record<string, string> = {
            'Question #': String(currentTestRun.questions.indexOf(question) + 1),
            'Question': question.question,
          };

          // Add captured groups
          for (let i = 0; i < columnNames.length && i < match.length - 1; i++) {
            row[columnNames[i]] = match[i + 1] || '';
          }

          extracted.push(row);
        }
      });

      if (extracted.length === 0) {
        setExtractionError('No matches found');
      } else {
        // Deduplicate if requested
        let finalData = extracted;
        if (extractUniqueOnly) {
          const seen = new Set<string>();
          finalData = extracted.filter(row => {
            // Create a key from all extracted columns (excluding Question # and Question)
            const key = columnNames.map(col => row[col] || '').join('|');
            if (seen.has(key)) {
              return false;
            }
            seen.add(key);
            return true;
          });
        }
        
        setExtractedData(finalData);
        setExtractionError('');
        setShowExtraction(true);
      }
    } catch (error) {
      setExtractionError(error instanceof Error ? error.message : 'Invalid regex pattern');
      setExtractedData([]);
    }
  };

  // Export extracted data to CSV
  const handleExportExtractedData = () => {
    if (extractedData.length === 0) return;

    // Get all column names
    const allColumns = Array.from(
      new Set(extractedData.flatMap(row => Object.keys(row)))
    );

    // Create CSV content
    const headers = allColumns.join(',');
    const rows = extractedData.map(row =>
      allColumns.map(col => {
        const value = row[col] || '';
        return `"${value.replace(/"/g, '""')}"`;
      }).join(',')
    );

    const csvContent = [headers, ...rows].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extracted_data_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-3 border-b border-white/10 backdrop-blur-xl bg-white/5 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Button
                variant={view === 'create' ? 'default' : 'outline'}
                size="sm"
                onClick={() => {
                  setView('create');
                  setCurrentTestRun(null);
                  setTestName('');
                  setPrompt('');
                  setQuestions('');
                }}
                className="text-xs"
              >
                New Test
              </Button>
              <Button
                variant={view === 'templates' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setView('templates')}
                className="text-xs"
              >
                Templates ({templates.length})
              </Button>
              <Button
                variant={view === 'history' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setView('history')}
                className="text-xs"
              >
                History ({testRuns.length})
              </Button>
            </div>
          </div>
          <Badge variant="secondary" className="text-xs">
            Model: {selectedModel}
          </Badge>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {view === 'create' && (
          <ScrollArea className="h-full p-6">
            <div className="max-w-3xl mx-auto space-y-4">
              <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
                <CardHeader>
                  <CardTitle className="text-lg">Create New Test Run</CardTitle>
                  <CardDescription>
                    Define a prompt template and list of questions to test
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-gray-700 mb-1.5 block">
                      Test Name
                    </label>
                    <input
                      type="text"
                      value={testName}
                      onChange={(e) => setTestName(e.target.value)}
                      placeholder="e.g., Math Addition Test"
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-gray-700 mb-1.5 block">
                      Prompt Template
                    </label>
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="e.g., Add the following two numbers"
                      rows={3}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      This prompt will be prepended to each question
                    </p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-gray-700 mb-1.5 block">
                      Questions (one per line)
                    </label>
                    <textarea
                      value={questions}
                      onChange={(e) => setQuestions(e.target.value)}
                      placeholder="2+4&#10;3+4&#10;5+6"
                      rows={10}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none font-mono"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      {questions.split('\n').filter(q => q.trim()).length} question(s)
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={handleCreateTest}
                      disabled={!testName.trim() || !prompt.trim() || !questions.trim()}
                      className="flex-1 bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] hover:to-[#47b9ff]/80 text-white"
                    >
                      Create Test Run
                    </Button>
                    <Button
                      onClick={handleSaveAsTemplate}
                      disabled={!testName.trim() || !prompt.trim() || !questions.trim()}
                      variant="outline"
                      className="text-xs"
                    >
                      Save as Template
                    </Button>
                  </div>
                  <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={runInParallel}
                        onChange={(e) => setRunInParallel(e.target.checked)}
                        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <div className="flex-1">
                        <span className="text-sm font-medium text-gray-700">
                          Run questions in parallel
                        </span>
                        <p className="text-xs text-gray-600 mt-0.5">
                          Execute all questions simultaneously for faster results (recommended for simple queries)
                        </p>
                      </div>
                    </label>
                  </div>
                </CardContent>
              </Card>
            </div>
          </ScrollArea>
        )}

        {view === 'results' && currentTestRun && (
          <ScrollArea className="h-full p-6">
            <div className="max-w-4xl mx-auto space-y-4">
              {/* Filter Card */}
              <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
                <CardContent className="p-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">
                      Filter Results (Regex)
                    </label>
                    <input
                      type="text"
                      value={filterText}
                      onChange={(e) => setFilterText(e.target.value)}
                      placeholder="e.g., Verdict:\s*(-?\d+) or error|fail"
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
                    />
                    {filterError && (
                      <p className="text-xs text-red-600">{filterError}</p>
                    )}
                    {filterRegex && !filterError && (
                      <p className="text-xs text-green-600">
                        ✓ Valid regex - Matching {currentTestRun.questions.filter(matchesFilter).length} of {currentTestRun.questions.length} results
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Extraction Card */}
              <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
                <CardContent className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-gray-700">
                        Extract Data (Regex with Capture Groups)
                      </label>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setShowExtraction(!showExtraction)}
                        className="text-xs"
                      >
                        {showExtraction ? '▼' : '▶'} {showExtraction ? 'Hide' : 'Show'}
                      </Button>
                    </div>
                    
                    {showExtraction && (
                      <>
                        <div className="space-y-2">
                          <label className="text-xs text-gray-600">
                            Regex Pattern (use parentheses for capture groups)
                          </label>
                          <input
                            type="text"
                            value={extractionPattern}
                            onChange={(e) => setExtractionPattern(e.target.value)}
                            placeholder="e.g., Verdict:\s*(-?\d+)|Temperature:\s*([0-9.]+)"
                            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
                          />
                        </div>

                        <div className="space-y-2">
                          <label className="text-xs text-gray-600">
                            Column Names (comma-separated, one per capture group)
                          </label>
                          <input
                            type="text"
                            value={extractionColumns}
                            onChange={(e) => setExtractionColumns(e.target.value)}
                            placeholder="e.g., Verdict, Temperature"
                            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>

                        {extractionError && (
                          <p className="text-xs text-red-600">{extractionError}</p>
                        )}

                        <div className="flex gap-2">
                          <Button
                            onClick={handleExtractData}
                            disabled={!extractionPattern.trim() || !extractionColumns.trim()}
                            size="sm"
                            className="bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] hover:to-[#47b9ff]/80 text-white"
                          >
                            Extract Data
                          </Button>
                          {extractedData.length > 0 && (
                            <>
                              <Badge variant="secondary" className="text-xs flex items-center">
                                {extractedData.length} row(s) extracted
                              </Badge>
                              <Button
                                onClick={handleExportExtractedData}
                                size="sm"
                                variant="outline"
                                className="text-xs"
                              >
                                Export to CSV
                              </Button>
                            </>
                          )}
                        </div>

                        {/* Extracted Data Table */}
                        {extractedData.length > 0 && (
                          <div className="mt-4 border border-gray-200 rounded-lg overflow-hidden">
                            <div className="max-h-60 overflow-auto">
                              <table className="w-full text-xs">
                                <thead className="bg-gray-100 sticky top-0">
                                  <tr>
                                    {Object.keys(extractedData[0]).map((col) => (
                                      <th key={col} className="px-3 py-2 text-left font-semibold text-gray-700 border-b">
                                        {col}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {extractedData.map((row, idx) => (
                                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                      {Object.keys(extractedData[0]).map((col) => (
                                        <td key={col} className="px-3 py-2 border-b border-gray-100">
                                          {row[col] || '-'}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        <div className="text-xs text-gray-500 bg-blue-50 p-2 rounded">
                          <strong>Example:</strong> To extract verdicts like "Verdict: -2", use pattern: <code className="bg-white px-1">Verdict:\s*(-?\d+)</code> with column name: <code className="bg-white px-1">Verdict</code>
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{currentTestRun.name}</CardTitle>
                      <CardDescription className="mt-1">
                        Prompt: {currentTestRun.prompt}
                      </CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={
                          currentTestRun.status === 'completed' ? 'default' :
                          currentTestRun.status === 'running' ? 'secondary' :
                          currentTestRun.status === 'error' ? 'destructive' :
                          'outline'
                        }
                      >
                        {currentTestRun.status}
                      </Badge>
                      {currentTestRun.status === 'completed' && (
                        <Button
                          size="sm"
                          onClick={() => handleExportCSV(currentTestRun)}
                          className="text-xs"
                        >
                          Export CSV
                        </Button>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-2">
                  {currentTestRun.status === 'draft' && (
                    <>
                      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={runInParallel}
                            onChange={(e) => setRunInParallel(e.target.checked)}
                            className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <div className="flex-1">
                            <span className="text-sm font-medium text-gray-700">
                              Run in parallel
                            </span>
                            <p className="text-xs text-gray-600 mt-0.5">
                              {runInParallel ? '⚡ All questions will run simultaneously' : '📝 Questions will run one at a time'}
                            </p>
                          </div>
                        </label>
                      </div>
                      <Button
                        onClick={handleRunTest}
                        disabled={isRunning}
                        className="w-full bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] hover:to-[#47b9ff]/80 text-white"
                      >
                        {isRunning ? 'Running...' : `Run Test (${currentTestRun.questions.length} questions)`}
                      </Button>
                    </>
                  )}

                  {isRunning && (
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <div className="flex items-center gap-2 text-sm text-blue-700">
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>
                          {runInParallel 
                            ? `Running ${runningQuestions.size} question(s) in parallel...`
                            : `Running question ${currentQuestionIndex + 1} of ${currentTestRun.questions.length}...`
                          }
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="space-y-3 mt-4">
                    {currentTestRun.questions.filter(matchesFilter).map((question) => {
                      const originalIdx = currentTestRun.questions.indexOf(question);
                      return (
                        <Card key={question.id} className="bg-white/60 border-gray-200/50">
                          <CardContent className="p-4">
                            <div className="space-y-2">
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                    <div className="text-xs font-medium text-gray-500">
                                      Question {originalIdx + 1}
                                      {question.duration && (
                                        <span className="ml-2 text-gray-400">
                                          ({question.duration.toFixed(2)}s)
                                        </span>
                                      )}
                                    </div>
                                    {question.toolCalls && question.toolCalls.length > 0 && (
                                      <Badge 
                                        variant="outline" 
                                        className="text-xs cursor-pointer hover:bg-gray-100"
                                        onClick={() => {
                                          if (question.toolCalls && question.toolCalls.length > 0) {
                                            handleToolClick(question.toolCalls[0]);
                                          }
                                        }}
                                      >
                                        {question.toolCalls.length} tool{question.toolCalls.length !== 1 ? 's' : ''}
                                      </Badge>
                                    )}
                                  </div>
                                  <div className="text-sm text-gray-800 font-mono bg-gray-50 p-2 rounded">
                                    {question.question}
                                  </div>
                                </div>
                                {isRunning && runningQuestions.has(originalIdx) && (
                                  <svg className="w-4 h-4 text-blue-500 animate-spin flex-shrink-0" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                  </svg>
                                )}
                              </div>

                            {question.answer && (
                              <div>
                                <div className="text-xs font-medium text-gray-500 mb-1">Answer</div>
                                <div className="text-sm text-gray-800 bg-green-50 p-2 rounded whitespace-pre-wrap">
                                  {question.answer}
                                </div>
                              </div>
                            )}

                            {question.error && (
                              <div>
                                <div className="text-xs font-medium text-red-500 mb-1">Error</div>
                                <div className="text-sm text-red-700 bg-red-50 p-2 rounded">
                                  {question.error}
                                </div>
                              </div>
                            )}

                              {(question.answer || question.error) && (
                                <div>
                                  <label className="text-xs font-medium text-gray-500 mb-1 block">
                                    Notes
                                  </label>
                                  <textarea
                                    value={question.notes || ''}
                                    onChange={(e) => handleUpdateNotes(question.id, e.target.value)}
                                    placeholder="Add notes about this result..."
                                    rows={2}
                                    className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
                                  />
                                </div>
                              )}

                              {/* Tool Calls List */}
                              {question.toolCalls && question.toolCalls.length > 0 && (
                                <div className="pt-2 border-t border-gray-200">
                                  <div className="text-xs font-medium text-gray-500 mb-1.5">
                                    Tool Calls ({question.toolCalls.length})
                                  </div>
                                  <div className="flex flex-wrap gap-1.5">
                                    {question.toolCalls.map((tool, toolIdx) => (
                                      <Button
                                        key={toolIdx}
                                        size="sm"
                                        variant="outline"
                                        onClick={() => handleToolClick(tool)}
                                        className="text-xs h-7 px-2"
                                      >
                                        {tool.name}
                                        {tool.duration > 0 && (
                                          <span className="ml-1 text-gray-400">
                                            ({tool.duration.toFixed(2)}s)
                                          </span>
                                        )}
                                      </Button>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>
          </ScrollArea>
        )}

        {view === 'templates' && (
          <ScrollArea className="h-full p-6">
            <div className="max-w-4xl mx-auto space-y-4">
              <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
                <CardHeader>
                  <CardTitle className="text-lg">Test Templates</CardTitle>
                  <CardDescription>
                    {templates.length} template(s) saved - Reusable test configurations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {templates.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <p className="text-sm">No templates yet</p>
                      <p className="text-xs mt-2 text-gray-400">
                        Create a test and click "Save as Template" to reuse it later
                      </p>
                      <Button
                        onClick={() => setView('create')}
                        className="mt-4"
                        size="sm"
                      >
                        Create Your First Template
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {templates.slice().reverse().map((template) => (
                        <Card key={template.id} className="bg-white/60 border-gray-200/50">
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <h4 className="text-sm font-semibold text-gray-800 mb-1">
                                  {template.name}
                                </h4>
                                <p className="text-xs text-gray-600 mb-2 line-clamp-2">
                                  {template.prompt}
                                </p>
                                <div className="flex items-center gap-4 text-xs text-gray-500">
                                  <span>{template.questions.length} question(s)</span>
                                  <span>{new Date(template.createdAt).toLocaleDateString()}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleLoadTemplate(template)}
                                  className="text-xs"
                                >
                                  Load
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => handleDeleteTemplate(template.id)}
                                  className="text-xs"
                                >
                                  Delete
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </ScrollArea>
        )}

        {view === 'history' && (
          <ScrollArea className="h-full p-6">
            <div className="max-w-4xl mx-auto space-y-4">
              <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
                <CardHeader>
                  <CardTitle className="text-lg">Previous Test Runs</CardTitle>
                  <CardDescription>
                    {testRuns.length} test run(s) saved
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {testRuns.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <p className="text-sm">No test runs yet</p>
                      <Button
                        onClick={() => setView('create')}
                        className="mt-4"
                        size="sm"
                      >
                        Create Your First Test
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {testRuns.slice().reverse().map((run) => (
                        <Card key={run.id} className="bg-white/60 border-gray-200/50">
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="text-sm font-semibold text-gray-800">{run.name}</h4>
                                  <Badge
                                    variant={
                                      run.status === 'completed' ? 'default' :
                                      run.status === 'running' ? 'secondary' :
                                      run.status === 'error' ? 'destructive' :
                                      'outline'
                                    }
                                    className="text-xs"
                                  >
                                    {run.status}
                                  </Badge>
                                </div>
                                <p className="text-xs text-gray-600 mb-2">{run.prompt}</p>
                                <div className="flex items-center gap-4 text-xs text-gray-500">
                                  <span>{run.questions.length} questions</span>
                                  <span>Model: {run.model}</span>
                                  <span>{new Date(run.createdAt).toLocaleDateString()}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleViewTestRun(run)}
                                  className="text-xs"
                                >
                                  View
                                </Button>
                                {run.status === 'completed' && (
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => handleExportCSV(run)}
                                    className="text-xs"
                                  >
                                    Export
                                  </Button>
                                )}
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => handleDeleteTestRun(run.id)}
                                  className="text-xs"
                                >
                                  Delete
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </ScrollArea>
        )}
      </div>

      {/* Tool Call Modal */}
      <ToolCallModal 
        toolCall={selectedToolCall} 
        open={isToolModalOpen} 
        onOpenChange={setIsToolModalOpen}
      />
    </div>
  );
}

