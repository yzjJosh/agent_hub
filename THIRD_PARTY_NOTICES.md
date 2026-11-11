# Third-party notices

The project's `AGPL-3.0-only` grant covers its original code. Imported material and separately installed components retain their own copyright and license terms. This document records known inclusions and important runtime exceptions, not a complete dependency inventory or clearance to distribute a Docker image. Preserve upstream notices when copying or distributing their material.

## reCognizer

The CAPTCHA implementation under `app/lib/recognizer/`, including its prompt handling, detection components, and Playwright challengers, contains code adapted from [Vinyzu/reCognizer](https://github.com/Vinyzu/recognizer). The 15 PNG fixtures under `tests/app/lib/recognizer/components/data/`, including `splitted/`, match upstream `tests/images/` assets at the comparison revision below.

**Copyright (C) 2024 Vinyzu. License: GNU GPL version 3 or, at your option, any later version (`GPL-3.0-or-later`).** The full [upstream license and grant](LICENSES/recognizer-GPL-3.0-or-later.txt) are retained verbatim. This project does not replace that grant with its original-code license.

**Modification and provenance notice, recorded 2026-11-08:** the imported source has been modified for integration into Agent Hub, including module organization, model loading, and asynchronous execution. The exact original import revision, import date, and dates of the individual source modifications are not established by the retained history. Revision `8a002c4574a1158ca4fa8519279427fce3614c97` was used for source and fixture comparison. It is not a claim that this was the imported revision. Compare, for example, the upstream [prompt handler](https://raw.githubusercontent.com/Vinyzu/recognizer/8a002c4574a1158ca4fa8519279427fce3614c97/recognizer/components/prompt_handler.py) and [license](https://raw.githubusercontent.com/Vinyzu/recognizer/8a002c4574a1158ca4fa8519279427fce3614c97/LICENSE).

GPLv3 section 13 permits combining GPLv3 code with AGPLv3 code while preserving the GPL terms for the GPL-covered portion. AGPLv3 section 13 applies its network-source requirements to the combination. See the retained [GPL text](LICENSES/recognizer-GPL-3.0-or-later.txt) and [AGPL text](LICENSE). The upstream grant is retained for imported fixtures to the extent licensed by their rightsholders. CAPTCHA imagery, branding, or other third-party content in those fixtures may have separate underlying rights, which this project does not expand.

## PDF.js and bundled core-js

The vendored files `app/console/static/js/pdfjs/pdf.min.mjs` and `pdf.worker.min.mjs` identify PDF.js **6.3.289**, build `1c8020a7d`, and carry **Copyright 2024 Mozilla Foundation**. PDF.js is licensed under [Apache 2.0](LICENSES/Apache-2.0.txt). Its embedded copyright and license headers are preserved. See the upstream `mozilla/pdf.js` [source entry point](https://raw.githubusercontent.com/mozilla/pdf.js/v6.3.289/src/pdf.js) and [LICENSE](https://raw.githubusercontent.com/mozilla/pdf.js/v6.3.289/LICENSE) at tag `v6.3.289`.

Both bundles also identify **core-js 3.50.0**, copyright 2013–2025 Denis Pushkarev and 2025–2026 CoreJS Company. Its [MIT license and notices](LICENSES/core-js-MIT.txt) are included verbatim from the [upstream license](https://raw.githubusercontent.com/zloirock/core-js/v3.50.0/LICENSE). These bundled notices remain applicable alongside the PDF.js license.

## Installed dependencies and browser binary

Dependencies keep the licenses and notices shipped with their installed versions. Important examples include:

| Component | Terms and distribution considerations |
| --- | --- |
| Ultralytics and ultralytics-thop | AGPL-3.0 terms apply to the open-source packages used for detection and profiling. Reference licenses: [Ultralytics v8.3.253](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.253/LICENSE) and [THOP v2.2.2](https://raw.githubusercontent.com/ultralytics/thop/v2.2.2/LICENSE). No commercial-license exception is supplied by this repository. |
| reverse-geocoder | Its [upstream license](https://raw.githubusercontent.com/thampiman/reverse-geocoder/master/LICENSE) is LGPL 2.1. The package's GeoNames-derived data has separate attribution requirements. Geographic data attribution: [GeoNames](https://www.geonames.org/), whose [data information](https://www.geonames.org/about.html) describes Creative Commons Attribution licensing. Retain the data notices supplied with the version you use. |
| CloakBrowser wrapper | [MIT, copyright 2026 CloakHQ](https://raw.githubusercontent.com/CloakHQ/CloakBrowser/v0.6.3/LICENSE). This license covers the wrapper, not the downloaded browser binary. |
| CloakBrowser Chromium binary | The [v0.6.3 binary license](https://raw.githubusercontent.com/CloakHQ/CloakBrowser/v0.6.3/BINARY-LICENSE.md) imposes separate proprietary restrictions. Dependency listings are expressly permitted. Public redistribution or bundling needs separate rights. Permitted internal use is distinguished from exposing browser functionality or control to third-party customers, which requires a separate OEM/SaaS agreement. The Dockerfile downloads the binary into the image. Review the terms for the binary version you actually obtain before distributing an image or offering hosted browser functionality. |

## Models, services, and runtime components

Model weights are not included in Git. Downloaded weights, datasets, and model services have their own licenses and access terms. Examples referenced by the code include CLIP, `CIDAS/clipseg-rd64-refined`, `Mozilla/docornot`, `PleIAs/OCRerrcr`, `duartebarbosadev/deep-image-orientation-detection`, and YOLO models. Consult each model's own repository and model card before use or redistribution. In particular, the [CLIPSeg source license](https://raw.githubusercontent.com/timojl/clipseg/master/LICENSE) explicitly excludes pretrained weights from its code license.

The Docker image also installs operating-system tools, browser components, and Python runtime dependencies. Their package licenses and notices remain applicable. Some PyTorch installations include NVIDIA runtime packages with proprietary terms. External APIs and configured services also retain their own terms. None of these components acquires an AGPL license merely by being installed or used with Agent Hub.
