# Third-party notices

## yingwang/xiangqi-bot

Source: https://github.com/yingwang/xiangqi-bot

Original checkout commit: `c1c984dac8e91c17fda0e234b588f689c93a1b91`.

Copyright (c) 2026 Ying Wang. MIT License; full text retained in `LICENSES/xiangqi-bot-MIT.txt`.

Derived components include `xiangqi_bot.py`, `xiangqi_cnn.py`, and the supplied `xiangqi_cnn.pt` model. Local changes include Windows integration, recognition/crop handling and validation. The upstream repository declares MIT for its code and supplies the CNN weights; this notice records that provenance without asserting a separate license grant from Tencent for screenshots or artwork.

## Pikafish

Source and license: https://github.com/official-pikafish/Pikafish

GPL v3. The engine is an external executable communicating via UCI. No engine executable or NNUE network is included in the Git source distribution. Obtain a matching executable/network release from upstream. If distributing an engine binary yourself, provide the corresponding source and required notices for that exact version under the GPL; a generic upstream link alone is not a replacement for those obligations.

## Other dependencies and images

Python packages listed in `requirements.txt` retain their respective licenses (including PyQt5's GPL/commercial licensing). They are installed separately and are not vendored here. The project's GPL-3.0-only license does not replace third-party notices.

README screenshots demonstrate the game interface; game trademarks, artwork and other third-party content remain with their respective owners.
