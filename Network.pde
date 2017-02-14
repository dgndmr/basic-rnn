class Network {
  ArrayList<Neuron> neurons;
  ArrayList<Connection> connections;

  float error;
  PVector position;
  float [] bias;
  int activeTrainingIndex=0;

  Network(float x, float y, float[] bias) {
    position = new PVector(x, y); //for drawing to scene
    neurons = new ArrayList<Neuron>();
    connections = new ArrayList<Connection>();
    this.bias = bias;
  }

  void addNeuron(Neuron n) {
    neurons.add(n);
  }

  void connect(Neuron prev, Neuron newNeuron, float weight) {
    Connection c = new Connection(prev, newNeuron, weight);
    newNeuron.addConnection(c);
    connections.add(c);
  } 

  void feedforward() {

    int lastNeuronIndex=0;

    //calculates only for hidden layers
    for (int hiddenLayerIndex=0; hiddenLayerIndex<hiddenLayerCount; hiddenLayerIndex++) {
      for (int hiddenLayerNeuronIndex=0; hiddenLayerNeuronIndex< neuronCountPerHiddenLayer; hiddenLayerNeuronIndex++) {
        lastNeuronIndex = inputCount + ((hiddenLayerIndex * neuronCountPerHiddenLayer) + hiddenLayerNeuronIndex);
        Neuron n = neurons.get( lastNeuronIndex );
        n.calculateOutput();
      }
    }

    //calculates only for output layer
    for (int outputNeuronIndex=0; outputNeuronIndex<outputCount; outputNeuronIndex++) {
      Neuron n = neurons.get( lastNeuronIndex + 1 + outputNeuronIndex );
      n.calculateOutput();
    }
  }

  void backPass() {
    //iterates from last neuron to first neuron
    for (int i=neurons.size()-1; i>= inputCount; i--) {
      Neuron n = neurons.get( i );
      n.backwardPass();
    }

    //after back propogation done => assignes newWeight to weight for each connection
    for (Connection c : connections) {
      c.weight = c.newWeight;
    }

    //set input values for next training set
    activeTrainingIndex++;
    if (activeTrainingIndex==trainingInputs.length) {
      activeTrainingIndex=0;
    }

    for (int i=0; i< getInputNeurons().size(); i++) {
      Neuron n = neurons.get( i );
      n.outputValue = trainingInputs[activeTrainingIndex][i];
    }

    //set output values for next training set
    int i=0;
    for (Neuron n : getOutputNeurons()) {
      //Neuron n = neurons.get( i );
      n.targetValue = trainingOutputs[activeTrainingIndex][i];
      i++;
    }
  }

  void calculateOutputErrors() {
    float errorTotal=0;
    for (int i=0; i< outputCount; i++) {
      Neuron n = neurons.get( (neurons.size()-1)-i );
      float diff = n.targetValue - n.outputValue;
      n.error = 0.5 * (diff * diff);
      errorTotal += n.error;
      //errorTotal += 0.5 * (diff * diff);
    }
    error = errorTotal;
    if (error==0) {
      zeroError=true;
    }
  }

  ArrayList<Neuron> getOutputNeurons() {
    ArrayList<Neuron> outputs = new ArrayList<Neuron>();
    for (int i=neurons.size()-outputCount; i<neurons.size(); i++) {
      Neuron n = neurons.get( i );
      outputs.add(n);
    }
    return outputs;
  }

  ArrayList<Neuron> getInputNeurons() {
    ArrayList<Neuron> inputs = new ArrayList<Neuron>();
    for (int i=0; i< inputCount; i++) {
      Neuron n = neurons.get( i );
      inputs.add(n);
    }
    return inputs;
  }

  void show() {
    pushMatrix();
    translate(position.x, position.y);

    for (Connection c : connections) {
      c.display();
    }

    for (Neuron n : neurons) {
      n.display();
    }

    popMatrix();
  }
}