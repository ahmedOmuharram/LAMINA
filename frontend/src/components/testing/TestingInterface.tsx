import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { apiClient } from '@/lib/api';
import type { TestRun } from '@/types/api';

interface TestingInterfaceProps {
  selectedModel: string;
}

export function TestingInterface({ selectedModel }: TestingInterfaceProps) {
  const [view, setView] = useState<'create' | 'results' | 'history'>('create');
  const [testName, setTestName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [questions, setQuestions] = useState('');
  const [currentTestRun, setCurrentTestRun] = useState<TestRun | null>(null);
  const [testRuns, setTestRuns] = useState<TestRun[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [runInParallel, setRunInParallel] = useState(false);
  const [runningQuestions, setRunningQuestions] = useState<Set<number>>(new Set());

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

  // Save test runs to localStorage
  const saveTestRuns = (runs: TestRun[]) => {
    setTestRuns(runs);
    localStorage.setItem('test_runs', JSON.stringify(runs));
  };

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

                  <Button
                    onClick={handleCreateTest}
                    disabled={!testName.trim() || !prompt.trim() || !questions.trim()}
                    className="w-full bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] hover:to-[#47b9ff]/80 text-white"
                  >
                    Create Test Run
                  </Button>
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
                    {currentTestRun.questions.map((question, idx) => (
                      <Card key={question.id} className="bg-white/60 border-gray-200/50">
                        <CardContent className="p-4">
                          <div className="space-y-2">
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1">
                                <div className="text-xs font-medium text-gray-500 mb-1">
                                  Question {idx + 1}
                                  {question.duration && (
                                    <span className="ml-2 text-gray-400">
                                      ({question.duration.toFixed(2)}s)
                                    </span>
                                  )}
                                </div>
                                <div className="text-sm text-gray-800 font-mono bg-gray-50 p-2 rounded">
                                  {question.question}
                                </div>
                              </div>
                              {isRunning && runningQuestions.has(idx) && (
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
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
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
    </div>
  );
}

