Network network;

int inputCount=5;      //total input neuron count 
int outputCount = 5;   //total output neuron count

int hiddenLayerCount =3;  //total hidden layer count
int neuronCountPerHiddenLayer = 8;  //neuron number count per hidden layer

float[] bias;            //layer bias values 
float[] inputValues;     //input neuron values
float[] outputTargets;   //output neuron values
float[] weights;         //connection weights

float[][] trainingInputs;
float[][] trainingOutputs;

float[] testingInput;
float[][] testingOutputs;
float[][] testingErrors;

float learningRate = .5;  //network's learning rate
int counter=0;            //iteration counter
boolean start=false;      //start/stop boolean
boolean zeroError=false;  //checks networks 100% success

boolean isTesting=false;  //if true=> "weights declared manually"; else => "weights will be randomly declared by connection count" 

float lineHeight = 15;    //for debug text's lines

Table table;              //for saving network values

Canvas nC; //network Canvas
Canvas sC; //stats Canvas

void setup() {
  size(1280, 720);
  //fullScreen();

  nC = new Canvas(0, 0, width, height-300, 30, color(240));
  sC = new Canvas(0, height-300, width, 300, 30, color(240));


  trainingInputs = new float[][]{
    {1, 0, 0, 0, 0}, 
    {0, 1, 0, 0, 0}, 
    {0, 0, 0, 1, 0}, 
    {0, 0, 0, 0, 1}, 
    {0, 0, 0, 0, 0}, 
  };

  trainingOutputs = new float[][]{
    {0, 0, 0, 0, 1}, 
    {0, 0, 0, 1, 0}, 
    {0, 1, 0, 0, 0}, 
    {1, 0, 0, 0, 0}, 
    {0, 0, 0, 0, 0}
  };

  testingInput = new float[]{
    1, 0, 0, 0, 0
  };

  bias = new float[]{1, 1, 1, 1, 1}; // hiddenLayerCount + 1

  setWeights();

  network = new Network(nC.w / 2, nC.h / 2, bias);

  int wIndex=0;  //for indexing connection weights

  //add inputs
  for (int i = 0; i < inputCount; i++) {
    float x = map(0, 0, max(1, hiddenLayerCount + 2), (nC.w/2*-1) + nC.marginLeft + nC.padding, (nC.w/2) - nC.marginRight - nC.padding);
    float y = map(i, 0, max(1, inputCount-1), (nC.h/2*-1) + nC.marginTop + nC.padding, (nC.h/2) - nC.marginBottom - nC.padding);

    Neuron n;
    if (isTesting) {
      // set input values from testingInput
      n = new Neuron(true, testingInput[i], false, 0, i, -1, x, y);
    } else {
      // set input values from trainingInputs
      n = new Neuron(true, trainingInputs[0][i], false, 0, i, -1, x, y);
    }
    network.addNeuron(n);
  }

  //add hidden layers
  for (int layer = 0; layer < hiddenLayerCount; layer++) {
    for (int j = 0; j < neuronCountPerHiddenLayer; j++) {

      float x = map(layer + 1, 0, max(1, hiddenLayerCount + 1), (nC.w/2*-1) + nC.marginLeft + nC.padding, (nC.w/2) - nC.marginRight - nC.padding);
      float y = map(j, 0, max(1, neuronCountPerHiddenLayer-1), (nC.h/2*-1) + nC.marginTop + nC.padding, (nC.h/2) - nC.marginBottom - nC.padding);

      Neuron n = new Neuron(j, layer, x, y);

      //if first hidden layer => connect hidden layer neurons with input neurons
      if (layer==0) {
        for (int inputLayerNeuronIndex=0; inputLayerNeuronIndex<inputCount; inputLayerNeuronIndex++) {
          Neuron inputNeuron = network.neurons.get(inputLayerNeuronIndex);
          network.connect(inputNeuron, n, weights[wIndex]);
          wIndex++;
        }
      } 
      //else => connect hidden layer neurons with previous layer neurons
      else {  
        for (int prevHiddenNeuronIndex=0; prevHiddenNeuronIndex<neuronCountPerHiddenLayer; prevHiddenNeuronIndex++) {
          int index = inputCount + ( ( neuronCountPerHiddenLayer * (layer-1) ) + prevHiddenNeuronIndex );
          Neuron prev = network.neurons.get( index );
          network.connect(prev, n, weights[wIndex]);
          wIndex++;
        }
      }

      network.addNeuron(n);
    }
  }

  //add output neurons to network and connect with previous hidden layers
  for (int outputNeuronIndex=0; outputNeuronIndex<outputCount; outputNeuronIndex++) {
    float x = map( hiddenLayerCount, 0, max(1, hiddenLayerCount), (nC.w/2*-1) + nC.marginLeft + nC.padding, (nC.w/2) - nC.marginRight - nC.padding);
    float y = map(outputNeuronIndex, 0, max(1, outputCount-1), (nC.h/2*-1) + nC.marginTop + nC.padding, (nC.h/2) - nC.marginBottom - nC.padding);

    Neuron output;
    if (isTesting) {
      // set output values -1 for testing
      output = new Neuron(false, 0, true, -1, outputNeuronIndex, hiddenLayerCount, x, y);
    } else {
      // set output values from trainingOutputs
      output = new Neuron(false, 0, true, trainingOutputs[0][outputNeuronIndex], outputNeuronIndex, hiddenLayerCount, x, y);
    }

    for (int prevHiddenNeuronIndex=0; prevHiddenNeuronIndex<neuronCountPerHiddenLayer; prevHiddenNeuronIndex++) {

      int index = inputCount + ( ( neuronCountPerHiddenLayer * (hiddenLayerCount-1) ) + prevHiddenNeuronIndex );

      Neuron prev = network.neurons.get( index );
      network.connect(prev, output, weights[wIndex]);
      wIndex++;
    }
    network.addNeuron(output);
  }

  testingOutputs = new float[trainingInputs.length][outputCount];
  for (int i=0; i<testingOutputs.length; i++) {
    for (int j=0; j<outputCount; j++) {
      testingOutputs[i][j] = 0.0;
    }
  }

  testingErrors = new float[trainingInputs.length][outputCount];
  for (int i=0; i<testingOutputs.length; i++) {
    for (int j=0; j<outputCount; j++) {
      testingErrors[i][j] = 0.0;
    }
  }

  frameRate(120);

  //for the first time feed forward the network
  network.feedforward();
  network.calculateOutputErrors();
}

void setWeights() {
  if (isTesting) {
    //declare weights manually

    weights = new float[]{2.4733543, -3.6720746, 0.7013865, -0.0967032, 0.83061993, -5.0543933, 3.7058225, 0.83350235, -5.0473905, 1.6810588, -0.61181027, 1.8573996, 0.23602289, 2.633681, -4.6055, 0.3917074, -1.13396, 0.6749166, 2.7005925, -4.1903043, 3.7465603, -5.452093, 0.58007026, -4.659097, 2.7338908, -3.1862464, -0.6961837, 0.94955313, 1.2723656, 2.343036, -3.6622536, 1.011802, 0.13629282, -4.9880466, 3.268871, -3.369364, 1.492624, 0.6526731, 2.586391, -2.7961943, 1.712724, -2.9328222, 0.9909854, 1.1620464, 0.03620271, -0.9212097, -0.59541976, -0.68948066, -1.6413412, 4.218808, -2.399316, -3.1996808, -2.9855957, 2.2835624, 2.3584719, -0.46363544, -2.3322482, -0.17416859, 2.109852, 1.0095828, -2.439022, -0.46027726, -1.4744315, 2.122411, -0.59458417, -0.42050093, 1.0507019, 1.2480507, -0.18180369, -0.24929914, -0.9188895, 0.79503876, 3.331428, -2.0724983, -1.2860456, -1.4623268, 2.5817745, -1.5373032, 2.6648283, -3.8556888, 0.562774, -0.117955714, -1.8036273, -2.1999648, -0.5534417, 1.4836078, -0.6301195, -0.94231933, -0.040111456, 2.3356338, -2.6511555, -3.3851733, -1.4168576, 1.5357584, 2.7425432, -0.33245602, -2.2033799, 0.37282178, 0.80915344, 1.15364, -1.3475809, 0.14990297, -2.5703068, 1.7338449, 1.0832968, -0.032693285, 1.0116036, 0.3902539, 0.8328886, 0.59519696, -0.17076467, 0.5848163, -2.908076, 4.5778046, -2.7615552, -0.6942861, -1.9788262, 3.9184043, 1.95362, 0.42843208, -1.412947, -0.5782116, 1.6883644, 0.07938623, -2.47174, -1.6330419, -0.62354225, 2.7803583, 0.8974487, 0.4712139, 1.1259606, 0.7464761, 0.59109724, -0.22246088, 0.38760072, 1.1264968, 3.0418246, -1.137102, -2.2953386, -0.85012615, 3.1227205, -0.5476623, 1.9731584, -5.3227158, 1.2054431, -1.2724459, 0.28334528, 0.04017185, 0.7595357, 0.21947722, -0.40778974, -0.030119292, -1.6029783, 2.5886743, -2.805536, -1.3905079, -1.4922357, 2.762389, 2.8710167, -0.12844878, 0.9354594, 0.6753651, 1.1337074, 0.8274013, 0.82367545, 0.895791, 0.72884256, 0.7057949, -2.790717, 3.8046353, -6.521133, -3.3374012, 4.711666, -2.2517638, 6.348132, -3.0170562, -0.09126787, -6.2309117, 3.8413706, 0.72095364, -8.42183, 0.34894493, -6.295149, 0.25374302, -1.9554508, -0.7630789, -1.2041649, -1.2385273, -1.213478, -1.9677533, -0.8232912, -1.8978282, -2.5350912, 8.460243, 1.964419, -1.9824814, -9.982826, -2.8713188, 2.2775688, -1.8837761, -0.64585036, -7.5104203, -3.0962737, -1.062286, 9.117945, -0.6380919, -4.8901715, -1.54668};
  } else {
    //declare weights randomly for each neuron

    //calculates weight count based on layers 
    int wCount = 
      (inputCount * neuronCountPerHiddenLayer ) + //for input connections
      (hiddenLayerCount * neuronCountPerHiddenLayer * neuronCountPerHiddenLayer) + // for hidden layer connections
      ( neuronCountPerHiddenLayer * outputCount) ;

    //(inputCount * (hiddenLayerCount * neuronCountPerHiddenLayer)) + 
    //((hiddenLayerCount * neuronCountPerHiddenLayer) * outputCount) ;
    weights = new float[wCount];
    for (int i=0; i<wCount; i++) {
      weights[i] = random(0, 1);
    }
  }
}

//saves network values to the text .csv file
void saveSystem() {
  table = new Table();

  for (int i=0; i<network.getInputNeurons().size(); i++) {
    table.addColumn("i" + i, Table.FLOAT);
  }
  for (int i=0; i<network.getOutputNeurons().size(); i++) {
    table.addColumn("ot" + i, Table.FLOAT); //output target
  }
  for (int i=0; i<bias.length; i++) {
    table.addColumn("b" + i, Table.FLOAT);
  }
  for (int i=0; i<network.connections.size(); i++) {
    table.addColumn("w" + i, Table.FLOAT);
  }


  TableRow row = table.addRow();

  for (int i=0; i<network.getInputNeurons().size(); i++) {
    Neuron n = network.getInputNeurons().get(i);
    row.setFloat("i" + i, n.outputValue);
  }
  for (int i=0; i<network.getOutputNeurons().size(); i++) {
    Neuron n = network.getOutputNeurons().get(i);
    row.setFloat("ot" + i, n.targetValue);
  }
  for (int i=0; i<bias.length; i++) {
    row.setFloat("b" + i, bias[i]);
  }
  for (int i=0; i<network.connections.size(); i++) {
    Connection c = network.connections.get(i);
    row.setFloat("w" + i, c.weight);
  }

  saveTable(table, "data/" + random(999999) + ".csv");
}

void mousePressed() {

  if (!start) {
    //starts network feedForward and back propogation methods
    start=true;
  } else {
    //stops network feedForward and back propogation methods
    start=false;
    saveSystem();
  }
}

void draw() {
  background(250);
  nC.show();
  sC.show();
  network.show();



  if (start && !zeroError && counter<10000000) {    
    for (int i=0; i<500; i++) {
      network.backPass();
      network.feedforward();
      network.calculateOutputErrors();

      for (int j=0; j<network.getOutputNeurons().size(); j++) {
        Neuron n = network.getOutputNeurons().get(j);
        testingOutputs[network.activeTrainingIndex][j] = n.outputValue;
      }

      counter++;
    }
  } else {
    for (int j=0; j<network.getOutputNeurons().size(); j++) {
      Neuron n = network.getOutputNeurons().get(j);
      testingOutputs[network.activeTrainingIndex][j] = n.outputValue;
    }
  }

  showInfo();
}

void showInfo() {
  fill(0);
  float x = sC.x + sC.marginLeft + 15;
  float firstLine =sC.y + sC.marginTop + lineHeight; 
  float y = firstLine;

  y+=lineHeight;

  text("training inputs", x, y);
  text("training targets", x + 150, y);
  text("training outputs", x + 300, y);

  //prints input/output values
  for (int i=0; i<trainingInputs.length; i++) {
    y+=lineHeight;

    text(trainingInputs[i][0] + " , " + trainingInputs[i][1] + " , " + trainingInputs[i][2] + " , " + trainingInputs[i][3] + " , " + trainingInputs[i][4], x, y);
    text(trainingOutputs[i][0] + " , " + trainingOutputs[i][1]+ " , " + trainingOutputs[i][2]+ " , " + trainingOutputs[i][3]+ " , " + trainingOutputs[i][4], x+150, y);
    for (int j=0; j<outputCount; j++) {
      text(String.format("%.6f", testingOutputs[i][j]) + "", x+300 + (j * 90), y );
    }
    for (int j=0; j<outputCount; j++) {
      text(String.format("%.6f", testingOutputs[i][j]) + "", x+300 + (j * 90), y );
    }
  }

  //prints iteration count
  y+=lineHeight*2;
  text("Iteration: " + counter, x, y);

  y+=lineHeight;
  text("error: " + String.format("%.6f", network.error), x, y);

  y+=lineHeight;
  text("Is Training: " + !isTesting, x, y);

  y+=lineHeight;
  text("FPS: " + String.format("%.0f", frameRate), x, y);
}