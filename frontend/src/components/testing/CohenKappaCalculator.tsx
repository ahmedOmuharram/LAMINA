import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface CohenKappaResult {
  kappa: number;
  agreement: string;
}

export function CohenKappaCalculator() {
  const [annotator1, setAnnotator1] = useState('');
  const [annotator2, setAnnotator2] = useState('');
  const [result, setResult] = useState<CohenKappaResult | null>(null);
  const [error, setError] = useState<string>('');
  const [isCalculating, setIsCalculating] = useState(false);

  const parseInput = (input: string): number[] => {
    return input
      .split('\n')
      .map(line => line.trim())
      .filter(line => line !== '')
      .map(line => parseInt(line, 10));
  };

  const handleCalculate = async () => {
    setError('');
    setResult(null);

    // Parse inputs
    const y1 = parseInput(annotator1);
    const y2 = parseInput(annotator2);

    // Validate inputs
    if (y1.length === 0 || y2.length === 0) {
      setError('Both annotators must have at least one value');
      return;
    }

    if (y1.length !== y2.length) {
      setError(`Arrays must have the same length. Annotator 1: ${y1.length}, Annotator 2: ${y2.length}`);
      return;
    }

    // Check for invalid values
    const validLabels = [-2, -1, 0, 1, 2];
    for (const val of y1) {
      if (isNaN(val) || !validLabels.includes(val)) {
        setError(`Invalid value in Annotator 1: ${val}. Must be one of: ${validLabels.join(', ')}`);
        return;
      }
    }
    for (const val of y2) {
      if (isNaN(val) || !validLabels.includes(val)) {
        setError(`Invalid value in Annotator 2: ${val}. Must be one of: ${validLabels.join(', ')}`);
        return;
      }
    }

    // Call API
    setIsCalculating(true);
    try {
      const response = await fetch(`${API_URL}/v1/cohen_kappa`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ y1, y2 }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to calculate Cohen\'s kappa');
        return;
      }

      const data: CohenKappaResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsCalculating(false);
    }
  };

  const handleClear = () => {
    setAnnotator1('');
    setAnnotator2('');
    setResult(null);
    setError('');
  };

  const getKappaBadgeVariant = (kappa: number) => {
    if (kappa < 0) return 'destructive';
    if (kappa < 0.4) return 'secondary';
    if (kappa < 0.6) return 'default';
    if (kappa < 0.8) return 'default';
    return 'default';
  };

  const annotator1Count = parseInput(annotator1).length;
  const annotator2Count = parseInput(annotator2).length;

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
        <CardHeader>
          <CardTitle className="text-lg">Cohen's Kappa Calculator</CardTitle>
          <CardDescription>
            Measure inter-annotator agreement between two raters using Cohen's kappa with quadratic weighting
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Annotator 1 */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 flex items-center justify-between">
                <span>Annotator 1</span>
                <Badge variant="outline" className="text-xs">
                  {annotator1Count} value(s)
                </Badge>
              </label>
              <Textarea
                value={annotator1}
                onChange={(e) => setAnnotator1(e.target.value)}
                placeholder="2&#10;1&#10;-2&#10;0&#10;1"
                className="font-mono text-sm h-64 resize-none"
              />
              <p className="text-xs text-gray-500">
                Enter one value per line. Valid values: -2, -1, 0, 1, 2
              </p>
            </div>

            {/* Annotator 2 */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 flex items-center justify-between">
                <span>Annotator 2</span>
                <Badge variant="outline" className="text-xs">
                  {annotator2Count} value(s)
                </Badge>
              </label>
              <Textarea
                value={annotator2}
                onChange={(e) => setAnnotator2(e.target.value)}
                placeholder="2&#10;1&#10;-1&#10;0&#10;2"
                className="font-mono text-sm h-64 resize-none"
              />
              <p className="text-xs text-gray-500">
                Enter one value per line. Must match Annotator 1 length
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={handleCalculate}
              disabled={isCalculating || !annotator1.trim() || !annotator2.trim()}
              className="flex-1 bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] hover:to-[#47b9ff]/80 text-white"
            >
              {isCalculating ? 'Calculating...' : 'Calculate Cohen\'s Kappa'}
            </Button>
            <Button
              onClick={handleClear}
              variant="outline"
              disabled={isCalculating}
            >
              Clear
            </Button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <Card className="bg-white/60 border-gray-200/50">
              <CardContent className="p-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">Cohen's Kappa (κ)</span>
                    <Badge variant={getKappaBadgeVariant(result.kappa)} className="text-lg px-3 py-1">
                      {result.kappa.toFixed(4)}
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">Agreement Level</span>
                    <Badge variant="secondary" className="text-sm">
                      {result.agreement}
                    </Badge>
                  </div>

                  <div className="pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-600 mb-2">
                      <strong>Interpretation Guidelines:</strong>
                    </p>
                    <ul className="text-xs text-gray-600 space-y-1 ml-4">
                      <li>• &lt; 0: Poor (worse than random)</li>
                      <li>• 0.00 - 0.20: Slight agreement</li>
                      <li>• 0.21 - 0.40: Fair agreement</li>
                      <li>• 0.41 - 0.60: Moderate agreement</li>
                      <li>• 0.61 - 0.80: Substantial agreement</li>
                      <li>• 0.81 - 1.00: Almost perfect agreement</li>
                    </ul>
                  </div>

                  <div className="pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500">
                      <strong>Note:</strong> This calculation uses quadratic weighting, which gives more credit
                      to near-misses than complete disagreements. The labels are assumed to be ordinal
                      [-2, -1, 0, 1, 2].
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Info Card */}
          <Card className="bg-blue-50/50 border-blue-200/50">
            <CardContent className="p-3">
              <p className="text-xs text-gray-700">
                <strong>About Cohen's Kappa:</strong> Cohen's kappa (κ) is a statistic that measures 
                inter-annotator agreement for categorical items. It takes into account the agreement 
                occurring by chance and is generally thought to be a more robust measure than simple 
                percent agreement calculation.
              </p>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
}

