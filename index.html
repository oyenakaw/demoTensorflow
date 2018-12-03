<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    .buttonArea {
      display: flex;
      margin:auto;
    }
    .buttonStyle {
      padding: 20px;
      margin: 30px;
    }
  </style>
  <title>Tutorial of TensorFlow.js</title>
</head>
<body>
  <div class="buttonArea">
    <input type="button" value="データ生成" onclick="act()" class="buttonStyle">
    <input type="button" value="学習" onclick="doTrain()" class="buttonStyle">
    <input type="button" value="評価更新" onclick="doUpdate()" class="buttonStyle">
  </div>
  <div class="buttonArea">
  </div>
  <div>
    <canvas id="canvas"></canvas>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.8.0"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.js"></script>
  <script>

    let tDataX;
    let tDataY;

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 5, inputShape: [1]}));
    model.add(tf.layers.dense({units: 5, activation: 'relu'}));
    model.add(tf.layers.dense({units: 5, activation: 'relu'}));
    model.add(tf.layers.dense({units: 5, activation: 'relu'}));
    model.add(tf.layers.dense({units: 5, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1, activation: 'linear'}));
    
    const learningRate = 0.00005;
    let optimizer = tf.train.sgd(learningRate);
    model.compile({optimizer: optimizer, loss: 'meanSquaredError'});

    window.chartColors = {
      red: "#FF0000",
      blue: "#0000FF"
    };

    /**
    /* プロット対象データクラス
     */
    let Model = class {
      constructor(_x, _y) {
        this.x = _x;
        this.y = _y;
      }
    };
    
    /**
    /* ランダムデータ生成処理
     */
    let genRandom = () => {
      let randomData = [];
      for(let i = 0; i < 100; i++){
        let xData = i * 0.1 - 5;
        let yData = (0.2 * Math.pow(xData, 3)) - (0.5 * Math.pow(xData, 2)) + (1.0 * xData) + 10 * Math.random();
        randomData[i] = new Model(xData, yData);
      }
      return randomData;
    };
    
    let genEstimatedValue = () => {
       let estimatedData = [];
       let xData = [];
       let yData = [];
       for (let i = 0; i < 100; i++){
         let xData = i * 0.1 - 5;
         let yData = doPredict(tf.tensor([xData], [1,1])).dataSync()[0];
         estimatedData[i] = new Model(xData, yData);
       }
       return estimatedData;
     };

    let doPredict = (x) => {
      return tf.tidy(() => {
        return model.predict(x);
      });
    }

    var color = Chart.helpers.color;
    var scatter_data = {
      datasets:[{
        label: "schatter dots",
        borderColor: window.chartColors.red,
        backgroundColor: color(window.chartColors.red).alpha(0.2).rgbString(),
        pointRadius: 10,

        data: genRandom(),
        type: 'scatter'
      },{
        label: "predict line",
        borderColor: window.chartColors.blue,
        backgroundColor: color(window.chartColors.blue).alpha(0.2).rgbString(),
        data: genEstimatedValue(),
        type: 'scatter'
      }]
    };

    let chart;
    let act = () => {
      var ctx = document.getElementById('canvas').getContext('2d');
      chart = new Chart(ctx, {
        type: 'scatter',
        data: scatter_data,
          option:{
            title: {
              display: true,
              text: "Chart.js Scatter Chart"
            },
          }
      });
    };

    let setData = () => {
      let x = [];
      let y = [];
      for(let i = 0; i < 100; i++) {
        x[i] = scatter_data.datasets[0].data[i].x;
        y[i] = scatter_data.datasets[0].data[i].y;
      }
      tDataX = tf.tensor1d(x);
      tDataY = tf.tensor1d(y);
    };

    let doTrain = () => {
      setData();
      model.fit(tDataX,tDataY,{epochs: 1000});
    };

    let doUpdate = () => {
      chart.data.datasets[1].data = genEstimatedValue();
      chart.update();
    };
  </script>
</body>
