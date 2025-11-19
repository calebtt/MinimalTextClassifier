using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using Serilog;

namespace MinimalTextClassifier.Core;

/// <summary>
/// General-purpose binary text classifier using a fine-tuned transformer model (e.g., DeBERTa or BERT variants) via ONNX Runtime.
/// Users are expected to provide a path to their fine-tuned ONNX model for the specific classification task.
/// Supports optional tokenizer model path (defaults to "models/spm.model" if not provided).
/// Download model files here: https://huggingface.co/microsoft/deberta-v3-small/tree/main
/// </summary>
public sealed class MinimalTransformerClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly long[] _inputIds;
    private readonly long[] _attentionMask;
    private bool _disposed;

    /// <summary>
    /// Initializes the classifier with the specified ONNX model path.
    /// </summary>
    /// <param name="modelPath">Path to the fine-tuned ONNX model file (required).</param>
    /// <param name="tokenizerPath">Optional path to the tokenizer model file (defaults to "models/spm.model").</param>
    public MinimalTransformerClassifier(string modelPath, string? tokenizerPath = null)
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        options.AppendExecutionProvider_CUDA();

        try
        {
            _session = new InferenceSession(modelPath, options);
            Log.Debug("Transformer model loaded from file: {Path}", modelPath);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or FileNotFoundException)
        {
            Log.Error(ex, "Failed to load transformer ONNX model");
            throw;
        }

        // Use default tokenizer path if not provided
        tokenizerPath ??= Path.Combine("models", "spm.model");
        if (!File.Exists(tokenizerPath))
            throw new FileNotFoundException("Tokenizer model file not found at " + tokenizerPath);
        using var tokenizerMemory = new MemoryStream(File.ReadAllBytes(tokenizerPath));
        _tokenizer = SentencePieceTokenizer.Create(tokenizerMemory);

        // Pre-allocate reusable buffers
        _inputIds = new long[128];
        _attentionMask = new long[128];

        Log.Information("MinimalTransformerClassifier initialized successfully with tokenizer");
    }

    /// <summary>
    /// Runs inference on the input text and returns the confidence score (0-1) for the positive class.
    /// The model should be fine-tuned for the specific binary classification task (e.g., via the provided Python script and example files).
    /// </summary>
    /// <param name="text">The text to classify.</param>
    /// <returns>The probability score for the positive class.</returns>
    public float ClassifyText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return 0f;

        text = text.Trim().ToLowerInvariant();

        // Use SentencePiece tokenizer (matches Python training)
        var ids = _tokenizer.EncodeToIds(text, addBeginningOfSentence: true, addEndOfSentence: true);
        var inputIdsList = ids.Select(i => (long)i).ToList();

        // Truncate to 128 if needed
        if (inputIdsList.Count > 128)
        {
            inputIdsList = inputIdsList.Take(128).ToList();
        }

        int len = inputIdsList.Count;
        for (int i = 0; i < len; i++)
        {
            _inputIds[i] = inputIdsList[i];
            _attentionMask[i] = 1;
        }
        for (int i = len; i < 128; i++)
        {
            _inputIds[i] = 0;       // PAD
            _attentionMask[i] = 0;
        }

        var inputTensor = TensorHelper.CreateTensor(_inputIds, new[] { 1, 128 });
        var maskTensor = TensorHelper.CreateTensor(_attentionMask, new[] { 1, 128 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor)
        };

        using var results = _session.Run(inputs);
        var logits = results[0].AsTensor<float>();
        float score = Softmax(logits.ToArray())[1]; // index 1 = positive class

        return score;
    }

    private static float[] Softmax(float[] x)
    {
        float max = x.Max();
        var exp = x.Select(v => MathF.Exp(v - max)).ToArray();
        float sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
    }
}

// Helper to avoid allocating new tensors every call
internal static class TensorHelper
{
    public static DenseTensor<long> CreateTensor(long[] data, int[] dimensions)
    {
        var tensor = new DenseTensor<long>(data, dimensions);
        return tensor;
    }
}