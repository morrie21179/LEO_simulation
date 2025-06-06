class View:
    def __init__(self, index: float, quality: float):
        self.index = index
        self.quality = quality
    def __call__(self):
        print("View",self.index, "Quality:", self.quality)  # Print output to console

def synthesized_view(v_L: View, v_R: View, index: float) -> View:

    new_quality = (v_L.quality + v_R.quality) / 2 - 0.1*(v_R.index - v_L.index)
    return View(index, new_quality)

    # v_L = View(0, 0.75)
    # v_R = View(2, 0.85)
    # new_view = synthesized_view(v_L, v_R, 1)
    # new_view()