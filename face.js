// 얼굴 엔진 — insightface 작은 모델(det_500m 얼굴 찾기 + w600k_mbf 특징값)을 브라우저(onnxruntime-web)에서 돌림.
// PC 사진 분류 앱(insightface buffalo_l)과 같은 방식을 그대로 옮김: 찾기 → 눈코입 5점으로 112×112 반듯하게 펴기 → 512개 특징값.
// 2026-10-01 검증(유년1부 아이 6명×2장): 파이썬 insightface와 특징값 유사도 0.94~0.999(큰 사진은 줄이는 방식 차이로 약간 낮음),
// 같은 아이 최저 0.481 / 다른 아이 최고 0.442 로 파이썬 작은 모델(0.481/0.439)과 같은 수준. 브라우저판 face-api·Human은 두 범위가 겹쳐 탈락.
(function (global) {
  var ORT_VER = '1.30.0';
  var ARC_DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
  var det = null, rec = null;

  function loadScript(src) {
    return new Promise(function (ok, fail) {
      var s = document.createElement('script');
      s.src = src; s.onload = ok;
      s.onerror = function () { fail(new Error('얼굴 인식 엔진을 못 불러왔어요(인터넷 확인)')); };
      document.head.appendChild(s);
    });
  }

  // modelBase: det_500m.onnx, w600k_mbf.onnx 가 있는 주소(끝에 /)
  function load(modelBase) {
    if (det && rec) return Promise.resolve();
    var need = global.ort ? Promise.resolve() :
      loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VER + '/dist/ort.min.js');
    return need.then(function () {
      global.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VER + '/dist/';
      var opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      return Promise.all([
        global.ort.InferenceSession.create(modelBase + 'det_500m.onnx', opt),
        global.ort.InferenceSession.create(modelBase + 'w600k_mbf.onnx', opt)
      ]);
    }).then(function (s) { det = s[0]; rec = s[1]; });
  }

  function canvas(w, h) {
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  // 그림 → RGB 입력값 [1,3,h,w], (픽셀-mean)/std
  function toTensor(ctx, w, h, mean, std) {
    var px = ctx.getImageData(0, 0, w, h).data, n = w * h;
    var out = new Float32Array(3 * n);
    for (var i = 0; i < n; i++) {
      out[i] = (px[i * 4] - mean) / std;
      out[n + i] = (px[i * 4 + 1] - mean) / std;
      out[2 * n + i] = (px[i * 4 + 2] - mean) / std;
    }
    return new global.ort.Tensor('float32', out, [1, 3, h, w]);
  }

  function iou(a, b) {
    var x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
    var inter = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
    var ar = function (r) { return (r[2] - r[0] + 1) * (r[3] - r[1] + 1); };
    return inter / (ar(a) + ar(b) - inter);
  }

  // 얼굴 찾기(SCRFD). 그림(sw×sh)을 size 정사각형 왼쪽 위에 맞춰 넣고, 결과 좌표는 원래 그림 기준.
  function detect(src, sw, sh, size, thresh) {
    size = size || 640; thresh = thresh || 0.5;
    var scale = Math.min(size / sw, size / sh);
    var c = canvas(size, size), ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, size, size);
    ctx.drawImage(src, 0, 0, Math.round(sw * scale), Math.round(sh * scale));
    var feeds = {}; feeds[det.inputNames[0]] = toTensor(ctx, size, size, 127.5, 128);
    return det.run(feeds).then(function (o) {
      var names = det.outputNames, strides = [8, 16, 32], faces = [];
      strides.forEach(function (st, k) {
        var sc = o[names[k]].data, bb = o[names[k + 3]].data, kp = o[names[k + 6]].data;
        var W = size / st;
        for (var i = 0; i < sc.length; i++) {
          if (sc[i] < thresh) continue;
          var cell = Math.floor(i / 2), cx = (cell % W) * st, cy = Math.floor(cell / W) * st;
          var box = [cx - bb[i * 4] * st, cy - bb[i * 4 + 1] * st, cx + bb[i * 4 + 2] * st, cy + bb[i * 4 + 3] * st]
            .map(function (v) { return v / scale; });
          var kps = [];
          for (var j = 0; j < 5; j++) kps.push([(cx + kp[i * 10 + j * 2] * st) / scale, (cy + kp[i * 10 + j * 2 + 1] * st) / scale]);
          faces.push({ box: box, score: sc[i], kps: kps });
        }
      });
      faces.sort(function (a, b) { return b.score - a.score; });
      var keep = [];
      faces.forEach(function (f) { if (keep.every(function (k) { return iou(k.box, f.box) <= 0.4; })) keep.push(f); });
      return keep;
    });
  }

  // 눈코입 5점 → 기준점으로 옮기는 닮음 변환(회전·크기·이동) 최소제곱
  function similarity(src) {
    var n = 5, ms = [0, 0], md = [0, 0];
    for (var i = 0; i < n; i++) { ms[0] += src[i][0] / n; ms[1] += src[i][1] / n; md[0] += ARC_DST[i][0] / n; md[1] += ARC_DST[i][1] / n; }
    var num1 = 0, num2 = 0, den = 0;
    for (var j = 0; j < n; j++) {
      var sx = src[j][0] - ms[0], sy = src[j][1] - ms[1], dx = ARC_DST[j][0] - md[0], dy = ARC_DST[j][1] - md[1];
      num1 += sx * dx + sy * dy; num2 += sx * dy - sy * dx; den += sx * sx + sy * sy;
    }
    var a = num1 / den, b = num2 / den;
    return [a, b, md[0] - (a * ms[0] - b * ms[1]), md[1] - (b * ms[0] + a * ms[1])];
  }

  // 얼굴 하나 → 512개 특징값(길이 1로 맞춤)
  function embed(src, face) {
    var t = similarity(face.kps), c = canvas(112, 112), ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 112, 112);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.setTransform(t[0], t[1], -t[1], t[0], t[2], t[3]);
    ctx.drawImage(src, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    var feeds = {}; feeds[rec.inputNames[0]] = toTensor(ctx, 112, 112, 127.5, 127.5);
    return rec.run(feeds).then(function (o) {
      var v = o[rec.outputNames[0]].data, s = 0;
      for (var i = 0; i < v.length; i++) s += v[i] * v[i];
      s = Math.sqrt(s);
      return Array.prototype.map.call(v, function (x) { return x / s; });
    });
  }

  // 그림에서 가장 큰 얼굴 하나의 특징값. 얼굴 없으면 null.
  function describe(src, sw, sh, size) {
    return detect(src, sw, sh, size).then(function (fs) {
      if (!fs.length) return null;
      var area = function (f) { return (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]); };
      var f = fs.reduce(function (a, b) { return area(b) > area(a) ? b : a; });
      return embed(src, f).then(function (e) { return { embedding: e, face: f }; });
    });
  }

  function cosine(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  global.FaceEngine = { load: load, detect: detect, embed: embed, describe: describe, cosine: cosine };
})(window);
