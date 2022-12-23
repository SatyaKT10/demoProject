if(window.innerWidth < 750){
  Webcam.set({
    width: window.innerWidth,
    height: (window.innerWidth *(4/3)),
    image_format: 'jpeg',
    flip_horiz: true,
    jpeg_quality: 100
  });
}
else if(window.innerWidth < 1200){
  Webcam.set({
    width: window.innerWidth/2,
    height: 450,
    dest_width: window.innerWidth/2,
    dest_height: 450,
    image_format: 'jpeg',
    flip_horiz: true,
    jpeg_quality: 100
  });
}
else{
  Webcam.set({
    width: 600,
    height: 450,
    dest_width: 600,
    dest_height: 450,
    image_format: 'jpeg',
    flip_horiz: true,
    jpeg_quality: 100
  });
}

Webcam.attach('#camera');

var currentPic = $(".color_icon-content:first").val();;
const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const WHITE = '#FFFFFF';

const silhouette = [
  [
      322.9636535644531,
      122.03394317626953,
      -4.161669731140137
  ],
  [
      360.002197265625,
      121.8666763305664,
      -3.259645462036133
  ],
  [
      391.5876159667969,
      126.0908432006836,
      0.984927773475647
  ],
  [
      420.749755859375,
      136.5352783203125,
      8.505210876464844
  ],
  [
      439.8554382324219,
      152.93128967285156,
      17.454593658447266
  ],
  [
      451.88470458984375,
      173.825927734375,
      25.945755004882812
  ],
  [
      458.2528381347656,
      196.22433471679688,
      34.93466567993164
  ],
  [
      460.6545715332031,
      224.31663513183594,
      41.953895568847656
  ],
  [
      459.7451171875,
      250.92405700683594,
      45.04695510864258
  ],
  [
      458.8947448730469,
      277.8260498046875,
      45.21182632446289
  ],
  [
      455.7530517578125,
      306.8681640625,
      43.17188262939453
  ],
  [
      448.0108337402344,
      337.7629089355469,
      38.72793960571289
  ],
  [
      436.6681213378906,
      363.0459899902344,
      31.50616455078125
  ],
  [
      423.57757568359375,
      382.2984619140625,
      23.636621475219727
  ],
  [
      406.4606628417969,
      399.91156005859375,
      17.24211311340332
  ],
  [
      391.2116394042969,
      412.41265869140625,
      12.309057235717773
  ],
  [
      374.640380859375,
      423.4629211425781,
      7.260157585144043
  ],
  [
      355.03179931640625,
      431.712646484375,
      3.9174981117248535
  ],
  [
      329.81036376953125,
      434.9303283691406,
      2.8289666175842285
  ],
  [
      304.764892578125,
      433.1107482910156,
      4.252836227416992
  ],
  [
      285.5359802246094,
      425.9767150878906,
      7.866791725158691
  ],
  [
      269.57891845703125,
      415.87298583984375,
      13.192002296447754
  ],
  [
      254.90093994140625,
      404.266845703125,
      18.27300262451172
  ],
  [
      238.27952575683594,
      387.8191223144531,
      24.843984603881836
  ],
  [
      225.614501953125,
      369.4324951171875,
      32.98933792114258
  ],
  [
      214.22653198242188,
      344.9697265625,
      40.440086364746094
  ],
  [
      205.51292419433594,
      314.76708984375,
      44.964378356933594
  ],
  [
      201.23402404785156,
      286.1849670410156,
      47.081748962402344
  ],
  [
      199.2126922607422,
      259.5066833496094,
      46.982906341552734
  ],
  [
      196.7713165283203,
      232.98342895507812,
      43.85902404785156
  ],
  [
      197.06936645507812,
      204.87779235839844,
      36.77418518066406
  ],
  [
      201.526123046875,
      182.19544982910156,
      27.674633026123047
  ],
  [
      211.5024871826172,
      160.70411682128906,
      18.96696662902832
  ],
  [
      228.44015502929688,
      143.18287658691406,
      9.799647331237793
  ],
  [
      255.85736083984375,
      130.8157501220703,
      1.900053858757019
  ],
  [
      286.3605651855469,
      124.39246368408203,
      -2.8329758644104004
  ]
];

async function predictions(image){
  await tf.setBackend('cpu');
  model = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
      {maxFaces: 1});
  const predictions = await model.estimateFaces({
      input: image,
      returnTensors: false,
      flipHorizontal: false,
      predictIrises: true
  });
  return predictions;
}

function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function drawLensToCanvas(prediction, base_image, ctx, brightness) {
  const keypoints = prediction.scaledMesh;
  if (keypoints.length > NUM_KEYPOINTS) {
      ctx.fillStyle = WHITE;
      ctx.lineWidth = 1;

      const leftCenter = keypoints[NUM_KEYPOINTS];
      // adding 5 to get better performance
      const leftDiameterY = distance(
          keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2])+ 3.5;
      const leftDiameterX = distance(
          keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1])+ 3.5;
      // console.log(leftDiameterX, leftDiameterY);
      // var ekR = ctx.getImageData(keypoints[NUM_KEYPOINTS + 3][0]+5,leftCenter[1],1,1).data;
      // var ekL = ctx.getImageData(keypoints[NUM_KEYPOINTS + 1][0]-5,leftCenter[1],1,1).data;
      // var hex1 = "#" + ("000000" + rgbToHex(ekR[0], ekR[1], ekR[2])).slice(-6);
      // var hex2 = "#" + ("000000" + rgbToHex(ekL[0], ekL[1], ekL[2])).slice(-6);
      // ctx.fillStyle = "#ff2626"; // Red color
      // ctx.beginPath(); //Start path
      // ctx.arc(keypoints[NUM_KEYPOINTS + 3][0]+5,leftCenter[1], 2, 0, Math.PI * 2, true);
      // ctx.arc(keypoints[NUM_KEYPOINTS + 1][0]-5,leftCenter[1], 2, 0, Math.PI * 2, true);
      // ctx.fill(); // Close the path and fill.

      // ctx.beginPath();
      // ctx.ellipse(
      //     leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2,
      //     0, 0, 2 * Math.PI);
      // ctx.stroke();
      // console.log(base_image);
      // Trial
      // Problem is non completion of a closed area
      // ctx.fillStyle = WHITE;
      lux0 = prediction.annotations.leftEyeUpper0[0][0];
      luy0 = prediction.annotations.leftEyeUpper0[0][1];

      lux1 = prediction.annotations.leftEyeUpper0[1][0];
      luy1 = prediction.annotations.leftEyeUpper0[1][1];

      lux2 = prediction.annotations.leftEyeUpper0[2][0];
      luy2 = prediction.annotations.leftEyeUpper0[2][1];

      lux3 = prediction.annotations.leftEyeUpper0[3][0];
      luy3 = prediction.annotations.leftEyeUpper0[3][1];

      lux4 = prediction.annotations.leftEyeUpper0[4][0];
      luy4 = prediction.annotations.leftEyeUpper0[4][1];

      lux5 = prediction.annotations.leftEyeUpper0[5][0];
      luy5 = prediction.annotations.leftEyeUpper0[5][1];

      lux6 = prediction.annotations.leftEyeUpper0[6][0];
      luy6 = prediction.annotations.leftEyeUpper0[6][1];

      llx0 = prediction.annotations.leftEyeLower0[0][0];
      lly0 = prediction.annotations.leftEyeLower0[0][1];

      llx1 = prediction.annotations.leftEyeLower0[1][0];
      lly1 = prediction.annotations.leftEyeLower0[1][1];

      llx2 = prediction.annotations.leftEyeLower0[2][0];
      lly2 = prediction.annotations.leftEyeLower0[2][1];

      llx3 = prediction.annotations.leftEyeLower0[3][0];
      lly3 = prediction.annotations.leftEyeLower0[3][1];

      llx4 = prediction.annotations.leftEyeLower0[4][0];
      lly4 = prediction.annotations.leftEyeLower0[4][1];

      llx5 = prediction.annotations.leftEyeLower0[5][0];
      lly5 = prediction.annotations.leftEyeLower0[5][1];

      llx6 = prediction.annotations.leftEyeLower0[6][0];
      lly6 = prediction.annotations.leftEyeLower0[6][1];

      llx7 = prediction.annotations.leftEyeLower0[7][0];
      lly7 = prediction.annotations.leftEyeLower0[7][1];

      llx8 = prediction.annotations.leftEyeLower0[8][0];
      lly8 = prediction.annotations.leftEyeLower0[8][1];

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(lux0, luy0);
      ctx.quadraticCurveTo(lux1, luy1, lux2, luy2);
      ctx.quadraticCurveTo(lux3, luy3, lux4, luy4);
      ctx.quadraticCurveTo(lux5, luy5, lux6, luy6);
      ctx.quadraticCurveTo(llx8, lly8, llx7, lly7);
      ctx.quadraticCurveTo(llx6, lly6, llx5, lly5);
      ctx.quadraticCurveTo(llx4, lly4, llx3, lly3);
      ctx.quadraticCurveTo(llx2, lly2, llx1, lly1);
      ctx.quadraticCurveTo(llx0, lly0, lux0, luy0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.clip();
      // ctx.fill();
      
      ctx.globalAlpha = brightness * 8/2000;
      // adding 3.5 to the left eye to get better performance is an option
      ctx.drawImage(base_image, leftCenter[0]-leftDiameterX/2, leftCenter[1]-leftDiameterY/2, leftDiameterX, leftDiameterY);
       //console.log(ctx);
      // Trial
      ctx.restore();

      if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          // adding 5 to get better performance
          const rightDiameterY = distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4])+ 3.5;
          const rightDiameterX = distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1])+ 3.5;

          // ctx.beginPath(); //Start path
          // ctx.arc(keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3][0]-5,rightCenter[1], 2, 0, Math.PI * 2, true);
          // ctx.arc(keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1][0]+5,rightCenter[1], 2, 0, Math.PI * 2, true);
          // ctx.fill(); // Close the path and fill.

          rux0 = prediction.annotations.rightEyeUpper0[0][0];
          ruy0 = prediction.annotations.rightEyeUpper0[0][1];

          rux1 = prediction.annotations.rightEyeUpper0[1][0];
          ruy1 = prediction.annotations.rightEyeUpper0[1][1];

          rux2 = prediction.annotations.rightEyeUpper0[2][0];
          ruy2 = prediction.annotations.rightEyeUpper0[2][1];

          rux3 = prediction.annotations.rightEyeUpper0[3][0];
          ruy3 = prediction.annotations.rightEyeUpper0[3][1];

          rux4 = prediction.annotations.rightEyeUpper0[4][0];
          ruy4 = prediction.annotations.rightEyeUpper0[4][1];

          rux5 = prediction.annotations.rightEyeUpper0[5][0];
          ruy5 = prediction.annotations.rightEyeUpper0[5][1];

          rux6 = prediction.annotations.rightEyeUpper0[6][0];
          ruy6 = prediction.annotations.rightEyeUpper0[6][1];

          rlx0 = prediction.annotations.rightEyeLower0[0][0];
          rly0 = prediction.annotations.rightEyeLower0[0][1];

          rlx1 = prediction.annotations.rightEyeLower0[1][0];
          rly1 = prediction.annotations.rightEyeLower0[1][1];

          rlx2 = prediction.annotations.rightEyeLower0[2][0];
          rly2 = prediction.annotations.rightEyeLower0[2][1];

          rlx3 = prediction.annotations.rightEyeLower0[3][0];
          rly3 = prediction.annotations.rightEyeLower0[3][1];

          rlx4 = prediction.annotations.rightEyeLower0[4][0];
          rly4 = prediction.annotations.rightEyeLower0[4][1];

          rlx5 = prediction.annotations.rightEyeLower0[5][0];
          rly5 = prediction.annotations.rightEyeLower0[5][1];

          rlx6 = prediction.annotations.rightEyeLower0[6][0];
          rly6 = prediction.annotations.rightEyeLower0[6][1];

          rlx7 = prediction.annotations.rightEyeLower0[7][0];
          rly7 = prediction.annotations.rightEyeLower0[7][1];

          rlx8 = prediction.annotations.rightEyeLower0[8][0];
          rly8 = prediction.annotations.rightEyeLower0[8][1];

          ctx.save();
          ctx.beginPath();
          ctx.moveTo(rux0, ruy0);
          ctx.quadraticCurveTo(rux1, ruy1, rux2, ruy2);
          ctx.quadraticCurveTo(rux3, ruy3, rux4, ruy4);
          ctx.quadraticCurveTo(rux5, ruy5, rux6, ruy6);
          ctx.quadraticCurveTo(rlx8, rly8, rlx7, rly7);
          ctx.quadraticCurveTo(rlx6, rly6, rlx5, rly5);
          ctx.quadraticCurveTo(rlx4, rly4, rlx3, rly3);
          ctx.quadraticCurveTo(rlx2, rly2, rlx1, rly1);
          ctx.quadraticCurveTo(rlx0, rly0, rux0, ruy0);
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = "high";
          ctx.clip();

          // ctx.fill();
          ctx.globalAlpha = brightness * 8/2000;
              // adding 3.5 to the right eye to get better performance is an option
          ctx.drawImage(base_image, rightCenter[0]-rightDiameterX/2, rightCenter[1]-rightDiameterY/2, rightDiameterX, rightDiameterY);

          ctx.restore();
      }
  }
}

// function rgbToHex(r, g, b) {
//     if (r > 255 || g > 255 || b > 255)
//         throw "Invalid color component";
//     return ((r << 16) | (g << 8) | b).toString(16);
// }

$(document).ready(function () {
  let dataColor = "Tricky Turquoise";
  let base_image = new Image();
  var canvas = document.getElementById("results");
  base_image.src = currentPic;
  base_image.crossOrigin="anonymous"
  
  let modelWidth = 600;
  let modelHeight = 450;
  if(window.innerWidth < 750){
    modelWidth = window.innerWidth;
    modelHeight = (window.innerWidth *(4/3));
  }
  else if(window.innerWidth < 1200){
    modelWidth = window.innerWidth/2;
    modelHeight = 450;
  }
  else{
    modelWidth = 600;
    modelHeight = 450;
  }

  document.getElementById("cameraWrapper").style.width = modelWidth+"px";
  document.getElementById("cameraWrapper").style.height = modelHeight+"px";
  
  
  canvas.width = modelWidth;
  canvas.height = modelHeight;
  var ctx = canvas.getContext("2d");
  var brightness = 0; 
  var prediction;
  var imageData;

  var overlayCanvas = document.getElementById("annotation");
  overlayCanvas.width = modelWidth;
  overlayCanvas.height = modelHeight;
  var overlayCtx = overlayCanvas.getContext("2d");

  var img = new Image();
  img.onload = function() {
     // overlayCtx.drawImage(img, 320-(img.width/4), 240-(img.height/4), img.width/2, img.height/2);
    overlayCtx.drawImage(img, (modelWidth/2)-(img.width/4), (modelHeight/2)-(img.height/4), img.width/2, img.height/2);
  }
  img.src = "https://cdn.shopify.com/s/files/1/0084/6957/7794/files/vector.svg";
  
  // for(var i=0; i<silhouette.length; i++){
  //     console.log(1);
  //     ctx.save();
  //     overlayCtx.fillStyle = "#ff2626"; // Red color
  //     overlayCtx.beginPath(); //Start path
  //     overlayCtx.arc(silhouette[i][0],silhouette[i][1], 2, 0, Math.PI * 2, true);
  //     overlayCtx.fill(); // Close the path and fill.
  //     ctx.restore();
  // }

  $('#lensify').click(function() {
    //$(this).hide();
    //$("#retryCappture").show();
    $(".model__info--wrapper").hide();
	$(".model__loader--wrapper").show();
    $('#lensify').prop('disabled', true);
    $('#retryCappture').prop('disabled', true);
    $('#retryCappture').removeClass("retry-active");
    $(".ar-page__capture-wrapper").addClass("hide");
    $(".ar-page__retry-wrapper").removeClass("hide");
    
//     base_image.src = 'https://cdn.shopify.com/s/files/1/0084/6957/7794/files/' + currentPic;


    Webcam.snap( function(data_uri) {
      
      
      //Added 04-03-2022
      var mainElement = document.getElementById("mainElement");
      mainElement.removeChild(mainElement.children[1]);
      canvas = document.createElement('canvas');
      canvas.width = modelWidth;
      canvas.height = modelHeight;
      canvas.id = 'results';
      mainElement.appendChild(canvas);
      ctx = canvas.getContext("2d");
      //Added 04-03-2022

      //console.log(data_uri.);
      
      var img = new Image();
      // create an image element for data-uri
      
      
      img.crossOrigin="anonymous";
      
      img.onload = async function() {           // add async handler        
        ctx.drawImage(this, 0, 0);      // when loaded, draw image (this)
        imageData = ctx.getImageData(0,0,canvas.width,canvas.height);
        // calcualting brightness
        var colorSum = 0;
        var data = imageData.data;
        var r,g,b,avg;
        for(var x = 0, len = data.length; x < len; x+=4) {
          r = data[x];
          g = data[x+1];
          b = data[x+2];
          avg = Math.floor((r+g+b)/3);
          colorSum += avg;
        }
        brightness = Math.floor(colorSum / (canvas.width*canvas.height));

        // ctx.putImageData(contrastImage(imageData, .98), 0, 0);
        prediction = await predictions(imageData);
        drawLensToCanvas(prediction[0], base_image, ctx, brightness);
        
       // ctx.putImageData(imageData, 0, 0);
        
        $('#lensify').prop('disabled', false);
        $('#retryCappture').prop('disabled', false);
        $('#retryCappture').addClass("retry-active");
        
        $(".model__action--wrapper").removeClass("hide");
        $(".model__loader--wrapper").hide();
        $("#colorIcon").removeClass("hide");
        saveModel();
        
      };
      
      img.src = data_uri; 
      
      document.getElementById("results").style.filter = "brightness(1.1)";
      document.getElementById("results").style.display = "block";

      $("#cameraWrapper").hide();
      Webcam.reset();
    });
  });
  
  $(".retry-active").on("click",function(){
    //$(this).hide();
    //$("#lensify").show();
    $(".ar-page__capture-wrapper").removeClass("hide");
    $(".ar-page__retry-wrapper").addClass("hide");
    $("#cameraWrapper").show();
    document.getElementById("results").style.display = "none";
    Webcam.attach( '#camera' );
    $('#retryCappture').prop('disabled', true);
    $('#retryCappture').removeClass("retry-active");
    $(".model__action--wrapper").addClass("hide");
    $("#colorIcon").addClass("hide");
    console.log("Webcam Attach")
  })

  $('.change').click(function() { 
    let $self = $(this);
    if(currentPic != this.value){
      currentPic = this.value;

      if(prediction !== undefined){
        base_image.src = currentPic;
        base_image.crossOrigin="anonymous"
        base_image.onload = function(){
          ctx.putImageData(imageData, 0, 0);
          drawLensToCanvas(prediction[0], base_image, ctx, brightness);
        }

        if(screen.width < 750){
          $('html, body').animate({
            scrollTop: $('#mainElement').offset().top - 70
          }, 500, 'linear');
        }
        
        $(".color_icon-content").removeClass("active");
        $(this).addClass("active");        
        dataColor = $($self).attr("data-color");
        //         clevertap events
        clevertap.event.push("favcolor", {
          'Name of the colors': dataColor
        });
        //           GA events
        let displayMode = 'browser';
        const mqStandAlone = '(display-mode: standalone)';
        if (navigator.standalone || window.matchMedia(mqStandAlone).matches || navigator.userAgent.toLowerCase().includes('wv')) {
          displayMode = 'standalone';
          dataLayer.push({
            'event': 'favcolor_pwa',
            'color_name': dataColor
          })
        }else{
          dataLayer.push({
            'event': 'favcolor',
            'color_name': dataColor
          })
        } 
        //          events end
        
      } else {
        console.log('no prediction');
      }

    }
  });


  async function shareCanvas() {
    var canvasElement = document.getElementById('results');
    var dataUrl = canvasElement.toDataURL();
    var blob = await (await fetch(dataUrl)).blob();
    var filesArray = [
      new File(
        [blob],
        'aqualens__'+new Date().getTime()+'.png',
        {
          type: blob.type,
          lastModified: new Date().getTime()
        }
      )
    ];
    var shareData = {
      files: filesArray,
    };
    try {
      await navigator.share(shareData)
    } catch (err) {
      if (err.name !== 'AbortError') {
        alert("Something went wrong!!");
      }
    }
  }
  
  function saveModel(){
//     let bearerToken = sessionStorage.getItem("token");
     let custNewToken =  sessionStorage.getItem("custNewToken");
    var canvasElement = document.getElementById('results');
    var dataUrl = canvasElement.toDataURL();

    var modelData ={
      image : dataUrl,
      predictions : prediction[0]
    }
    var settings = {
//    "url": baseUrl+"api/v1/ar/images",
      "url": baseURL+"api/v1/ar/images",
      "method": "POST",
      "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer "+custNewToken
      },
      dataType: "json",
      data: modelData
    };

    $.ajax(settings).done(function (response) {
      console.log(response);
    }).fail(function (error) {
      console.log("Error",error);
    })
    // clevertap events
    clevertap.event.push("favcolor", {
      'Name of the colors': dataColor
    });
    //           GA events
    let displayMode = 'browser';
    const mqStandAlone = '(display-mode: standalone)';
    if (navigator.standalone || window.matchMedia(mqStandAlone).matches || navigator.userAgent.toLowerCase().includes('wv')) {
      displayMode = 'standalone';
      dataLayer.push({
        'event': 'favcolor_pwa',
      })
    }else{
      dataLayer.push({
        'event': 'favcolor',
      })
    } 
    //          events end

  }

  $(".model__share--wrapper").on('click', function () {
//    shareCanvas()
    if (navigator.share) { 
      shareCanvas()
    }
    else {
		$(".share-popup__container").show();
        $("body").css({"overflow":"hidden"}); 
        $(document).mouseup(function (e) {
            if ($(e.target).closest(".share-popup__container").length === 0) {
                $(".share-popup__container").hide();
                $("body").css({"overflow":"auto"}); 
            }
        });
    }
  })
  
  $(".model__like--wrapper").on('click', function () {
    saveModel();
    $(this).find('svg').toggleClass("svg__active");
  })

  $(document).on("click",".share-popup__close, .share__button",function(){
    $(".share-popup__container").hide();
    $("body").css({"overflow":"auto"}); 
  })
  
  $(document).on("click",".model__info--wrapper",function(){
    $(this).hide(); 
  })
  
});

// document.getElementById("change").addEventListener("click", changeImage);
// function changeImage() {
//     if(currentPic == "dot2.png"){
//         currentPic = 'dot1.png';
//     } else {
//         currentPic = 'dot2.png';
//     }
// }

function contrastImage(imgData, contrast){  //input range [-100..100]
  var d = imgData.data;
  contrast = (contrast/100) + 1;  //convert to decimal & shift range: [0..2]
  var intercept = 128 * (1 - contrast);
  for(var i=0;i<d.length;i+=4){   //r,g,b,a
      d[i] = d[i]*contrast + intercept;
      d[i+1] = d[i+1]*contrast + intercept;
      d[i+2] = d[i+2]*contrast + intercept;
  }
  return imgData;
}