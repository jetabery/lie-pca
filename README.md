# lie-pca

'get_periodic_orbits.py' uses code from <https://github.com/HLovisiEnnes/LieDetect>

# Recovering Lie Groups from Orbits

## Author
**Julius Tabery**  
*Date: September 2024*

---

## 1. Best-Case Scenario
Suppose \( Q \in U(d) \) and \( a \in \mathbb{Z}^d \) (non-zero vector). The Lie group

\[
G = \{ Q e^{\text{diag}(2\pi i a t)} Q^{-1} : t \in [0, 1) \}
\]

represents the 1-torus \( T \). We identify \( T \) with the interval \( [0, 1) \) with addition modulo 1. Define

\[
G(t) = Q e^{\text{diag}(2\pi i a t)} Q^{-1}
\]

for \( t \in T \cong \mathbb{R}/\mathbb{Z} \). Assume that the map \( t \mapsto G(t)x \) is injective for a generic \( x \in \mathbb{C}^d \) (TODO: prove this holds generically).

### Recovery Process:
1. Given points \( x_1, \ldots, x_n \in \mathbb{C}^d \) where:

\[
x_j = G(j/n)x \quad \text{for } j \in [n]
\]

2. For each \( b \in \mathbb{Z} \), compute:

\[
y_b = \sum_{j=1}^n e^{-2\pi i b j/n} x_j
\]

3. Then:

\[
y_b = n P_b x
\]

where \( P_b \) is the orthogonal projector onto the subrepresentation of \( G \) corresponding to \( b \). If \( y_b \neq 0 \), then:

\[
P_b = \frac{y_b y_b^*}{\|y_b\|^2}
\]

4. Define:

\[
H(t) = I + \sum_{b \in \mathbb{Z}, y_b \neq 0} \frac{y_b y_b^*}{\|y_b\|^2} (e^{2\pi i b t} - 1)
\]

5. Therefore:

\[
H(t) x = G(t) x
\]

Thus, the simplest Lie group \( H \) that generates the orbit is:

\[
H = \{ H(t) : t \in [0, 1) \}
\]

If all nontrivial subrepresentations are 1-dimensional, then \( H = G \).

---

## 2. What if the Dataset is Reordered/Unordered?
Let \( D_n \) be the dihedral group as a subgroup of the symmetric group \( S_n \). 

### Theorem 1
*Suppose \( H_1 \) is calculated from the ordered dataset \( x_1, \ldots, x_n \). If the dataset is reordered by some permutation \( \sigma \in D_n \), then the recovered Lie group \( H_2 \) is identical to \( H_1 \).*

**Proof Outline**:
- Case 1: \( \sigma \) is a rotation → phase shift:

\[
y_b^{(2)} = e^{2\pi i b k/n} y_b^{(1)}
\]

- Case 2: \( \sigma \) is a reflection → conjugation by inversion:

\[
y_b^{(2)} = e^{-2\pi i b/n} y_{-b}^{(1)}
\]

Thus, in both cases, \( H_1 = H_2 \).

---

## 3. What if the Points are Not Evenly Spaced?
- Let \( t_1, \ldots, t_n \) be sampled uniformly from \( T \).
- Construct the matrix:

\[
K_{ij} = \exp \left( -\frac{\|x_{t_i} - x_{t_j}\|^2}{2 \epsilon^2} \right)
\]

- Perform Sinkhorn's algorithm and proceed as in the evenly spaced case.

---

## 4. What if We Add Noise?
(TODO: Prove this holds with high probability under noise.)

---

## 5. Technical Lemmas
### Lemma 1
Let \( f : T \to \mathbb{R} \) be differentiable, with \( f'(0) \neq 0 \) and \( f^{-1}(0) = \{0\} \). Then:

\[
\lim_{n \to \infty} \int_{-n/2}^{n/2} \left| e^{-\frac{n^2 f(t/n)^2}{f'(0)^2}} - e^{-t^2} \right| dt = 0
\]

(TODO: Clean up the proof.)

### Lemma 2
For sufficiently large \( n \), the eigenvectors of the kernel matrix:

\[
K_{ij} = \exp \left( -\frac{\|x_i - x_j\|^2}{2 \epsilon^2} \right)
\]

correspond to orthogonal sinusoids of frequency 1.

---

## 6. Practical Notes
- The method generalizes to higher-dimensional Lie groups.
- Next steps:
  - Prove that the result holds under noise.
  - Extend to higher-dimensional orbits.
  - Explore applications in machine learning and data augmentation.

---

## 🚀 TODO:
- Prove that \( H \) is a Lie group.
- Reformulate in terms of discrete Fourier transform.
- Clean up proofs and improve clarity.

---

## 📜 License
MIT License
