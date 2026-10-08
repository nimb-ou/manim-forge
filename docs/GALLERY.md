# Gallery

Ten scenes made by v1.0 exactly as the app makes them: Qwen3.5-9B untuned on the Mac (MLX 4-bit),
the two nearest library scenes, the kit reference, the checks, best of 2, Kokoro narration, 720p.
Rendered 2026-10-08 by `scripts/make_gallery.py`; every video has its narration track.

**How they were chosen, honestly:** 14 prompts were run. 8 were right by eye on the first pass.
The 6 that were not were re-run once from sampled answers (best of 3); 2 of those came out right
(06 and 09). The 4 that never did are listed at the end. Nothing was edited by hand.

| # | request | what it shows | time to make |
|---|---|---|---|
| [01](gallery/01-a-jacket-costs-80-and-is-15-off-what-do.mp4) | a jacket costs £80 and is 15% off. what do you pay? | £80 as a bar, the £12 cut, £68 left; 80 × 0.85 = 68 | 47 s |
| [02](gallery/02-if-9-people-all-shake-hands-with-each-ot.mp4) | if 9 people all shake hands with each other, how many handshakes is that? | nine people, A's eight handshakes, all 36 edges; 8 + 7 + … + 1 = 9 × 8 / 2 = 36 | 53 s |
| [03](gallery/03-why-do-the-angles-of-a-hexagon-add-up-to.mp4) | why do the angles of a hexagon add up to 720 degrees? | the hexagon cut into four triangles from one corner; 4 × 180° = 720°, 120° each | 40 s |
| [04](gallery/04-how-do-you-add-3-4-and-1-6.mp4) | how do you add 3/4 and 1/6? | 3/4 and 1/6 as bars, both cut into twelfths; 9/12 + 2/12 = 11/12 | 107 s |
| [05](gallery/05-find-the-area-under-y-3x-from-0-to-4.mp4) | find the area under y = 3x from 0 to 4 | the triangle under y = 3x, rectangles closing in on it; ½ · 4 · 12 = 24 | 84 s |
| [06](gallery/06-what-does-multiplying-by-i-do-to-the-poi.mp4) | what does multiplying by i do to the point 2 + i? | 2 + i turned a quarter turn to −1 + 2i on the complex plane | 43 s (retry) |
| [07](gallery/07-what-s-the-chance-of-two-heads-when-you.mp4) | what's the chance of two heads when you flip two coins? | the tree of two flips, HH one of four outcomes; P = 1/4 | 65 s |
| [08](gallery/08-solve-4x-3-19-using-a-balance.mp4) | solve 4x + 3 = 19 using a balance | a balance with four x-blocks and 3 weights against 19; 4x = 16, x = 4 | 92 s |
| [09](gallery/09-why-is-the-area-of-a-circle-pi-r-squared.mp4) | why is the area of a circle pi r squared? | a circle in 8 slices, laid out bumpy, then 32 slices: a rectangle πr wide and r tall; A = πr² | 80 s (retry) |
| [11](gallery/11-the-derivative-as-the-slope-of-the-tange.mp4) | the derivative as the slope of the tangent line | a parabola with its tangent sliding over the top: slope 0 at the peak, negative after | 77 s |

Contact sheets (the last frame of each beat):

### 01 · a jacket costs £80 and is 15% off. what do you pay?

![01-a-jacket-costs-80-and-is-15-off-what-do](gallery/01-a-jacket-costs-80-and-is-15-off-what-do.jpg)

### 02 · if 9 people all shake hands with each other, how many handshakes is that?

![02-if-9-people-all-shake-hands-with-each-ot](gallery/02-if-9-people-all-shake-hands-with-each-ot.jpg)

### 03 · why do the angles of a hexagon add up to 720 degrees?

![03-why-do-the-angles-of-a-hexagon-add-up-to](gallery/03-why-do-the-angles-of-a-hexagon-add-up-to.jpg)

### 04 · how do you add 3/4 and 1/6?

![04-how-do-you-add-3-4-and-1-6](gallery/04-how-do-you-add-3-4-and-1-6.jpg)

### 05 · find the area under y = 3x from 0 to 4

![05-find-the-area-under-y-3x-from-0-to-4](gallery/05-find-the-area-under-y-3x-from-0-to-4.jpg)

### 06 · what does multiplying by i do to the point 2 + i?

![06-what-does-multiplying-by-i-do-to-the-poi](gallery/06-what-does-multiplying-by-i-do-to-the-poi.jpg)

### 07 · what's the chance of two heads when you flip two coins?

![07-what-s-the-chance-of-two-heads-when-you](gallery/07-what-s-the-chance-of-two-heads-when-you.jpg)

### 08 · solve 4x + 3 = 19 using a balance

![08-solve-4x-3-19-using-a-balance](gallery/08-solve-4x-3-19-using-a-balance.jpg)

### 09 · why is the area of a circle pi r squared?

![09-why-is-the-area-of-a-circle-pi-r-squared](gallery/09-why-is-the-area-of-a-circle-pi-r-squared.jpg)

### 11 · the derivative as the slope of the tangent line

![11-the-derivative-as-the-slope-of-the-tange](gallery/11-the-derivative-as-the-slope-of-the-tange.jpg)

## Not good enough

Sheets in `gallery/rejected/`.

- **10**: what does a 2x2 matrix do to the plane? — the area scaling by det 2.5 is right, but the captions name the wrong columns and run together
- **12**: how far does a bike wheel with a 35 cm radius roll in one turn? — 2.2 m is right; the roll is barely drawn
- **13**: fitting a straight line to data with least squares — first try a muddled curve; retry draws bars, never the fitted line
- **14**: why does a stretched spring pull back harder the further you pull it? — mostly empty frames, then an unasked energy sum
