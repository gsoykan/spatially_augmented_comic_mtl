from typing import Optional


class TextInPanel:
    def __init__(self,
                 text: str,
                 x1: Optional[int] = None,
                 y1: Optional[int] = None,
                 x2: Optional[int] = None,
                 y2: Optional[int] = None):
        self.text = text
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __repr__(self):
        return self.text
