1. Thoughts on `Integer` vs `Int`?
2. Thoughts on typing for B in compress (AbstractArray, Matrix, Vector)?
3. Elegant way to do matrix multiplication with 1-d Matrix structs? (it seems clunky to define a compress method for a Matrix and Vector, especially since adjoints aren't even a subtype of matrix) --- should I reinterpret vector as matrix?
4. Is there a way to make pca compressor immutable?