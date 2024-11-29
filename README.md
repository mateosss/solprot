# Solve patch rotation

Given two observations of the same feature (from two different images), solve
the relative 2D rotation from one observation to the next.

![Demo video](docs/demo.gif)

# Requirements

- `pip install -r requirements.txt`
- Update Makefile if needed
- `make`
- `python main.py`

# Minimum example

While `main.py` has been used as a playground to try different optimization
methods. A simplified script with the currently preferred method is
`solprot.py`. You can use it like this:

```py
import solprot
angle = solprot.solve_patch_rotation("images/MOO11A.png", "images/MOO11B.png", [106.243, 331.75], [599.882, 313.076])
print(f"{angle=}")
```
