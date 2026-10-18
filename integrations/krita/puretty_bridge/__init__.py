from krita import Krita

from .bridge import PurettyBridgeExtension

Krita.instance().addExtension(PurettyBridgeExtension(Krita.instance()))
