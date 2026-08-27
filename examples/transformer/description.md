# Intended diagram: Transformer encoder architecture

Create a clear, top-to-bottom architecture diagram for a Transformer encoder.

The diagram must contain these components and meaningful labels:

1. **Input Tokens** at the top.
2. Two parallel components beneath the input: **Token Embeddings** and **Positional Encoding**.
3. An **Add** operation that combines Token Embeddings with Positional Encoding.
4. A visibly grouped **Transformer Encoder Block**, marked **N×** to show repetition.
5. Inside that block, in order:
   - **Multi-Head Self-Attention**
   - **Add & Norm**
   - **Position-wise Feed-Forward Network**
   - **Add & Norm**
6. **Contextual Embeddings** as the output beneath the encoder block.

The directed connectivity must be visible:

- Input Tokens → Token Embeddings.
- Token Embeddings → Add.
- Positional Encoding → Add.
- Add → Multi-Head Self-Attention.
- Multi-Head Self-Attention → the first Add & Norm.
- A residual/skip connection from Add → the first Add & Norm.
- The first Add & Norm → Position-wise Feed-Forward Network.
- Position-wise Feed-Forward Network → the second Add & Norm.
- A residual/skip connection from the first Add & Norm → the second Add & Norm.
- The second Add & Norm → Contextual Embeddings.

Layout requirements:

- Use a sensible top-to-bottom reading order.
- Keep all content inside the canvas.
- Avoid unintended overlaps.
- Make the repeated encoder block visually distinct from its internal components.
- Route arrows so their source and target are visually unambiguous.

Legibility requirements:

- Use meaningful labels rather than placeholders such as “Layer”, “Thing”, “TBD”, or “???”.
- All text must be readable, remain inside its intended box, and use a rendered font size of at least 12 px.

