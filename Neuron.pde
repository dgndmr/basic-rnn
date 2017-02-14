/*
 *  forwardpassing and back propogation values
 *  calculated based on
 *  https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *  (sorry for long variable names)
 */

class Neuron {
  PVector position;
  int layerIndex;        //at which layer is this neuron. "-1" for input layer
  int layerNeuronIndex;  //index of neuron on layer

  ArrayList<Connection> connections;

  boolean isOutput; //is this neuron output?
  float targetValue; //if this neuron is output => keeps target value

  boolean isInput;  //is this neuron input?

  float outputValue;  
  float error;

  float errorDivOut;
  float outDivInput;

  //constructor for input/output neuron
  Neuron(boolean isInput, float input, boolean isOutput, float target, int index, int LIndex, float x, float y) {

    position = new PVector(x, y);
    connections = new ArrayList<Connection>();
    this.isOutput = isOutput;
    this.targetValue = target;

    this.isInput = isInput;
    this.outputValue = input;

    this.layerNeuronIndex = index;
    this.layerIndex=LIndex;
  }

  //constructor for hidden neuron
  Neuron(int index, int LIndex, float x, float y) {
    position = new PVector(x, y);
    connections = new ArrayList<Connection>();
    this.isOutput = false;
    this.targetValue = 0;

    this.isInput = false;
    this.outputValue = 0;

    this.layerNeuronIndex = index;
    this.layerIndex=LIndex;
  }

  void addConnection(Connection c) {
    connections.add(c);
  }

  // calculates neurons output value.
  void calculateOutput() {
    float score =0;
    for (Connection c : connections) {
      score += c.weight * c.from.outputValue;
    }
    
    score = score + network.bias[ this.layerIndex ];
    this.outputValue = 1/(1+ exp(score*-1));   //keeps calculated value 
    
  }

  void backwardPass() {
    /*
     * at https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
     * there are simple different calculations for output and hidden layer back propogation
     * 
     */
    
    if (isOutput) {
      /*
       * this is only for output neurons
       * you could find at "The Backwards Pass => Output Layer" section
       * on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
       */
      errorDivOut = (this.targetValue - this.outputValue) * -1;
      outDivInput = this.outputValue * (1 - this.outputValue);

      for (Connection c : connections) {
        float inputDivW = c.from.outputValue;
        float errorTotalDivW = errorDivOut * outDivInput * inputDivW;
        c.newWeight = c.newWeight - learningRate * errorTotalDivW;
      }
    } else if (!isOutput && !isInput) {
      
      /*
       * this is only for hidden layer neurons
       * you could find at "The Backwards Pass => Hidden Layer" section
       * on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
       */
       
      ArrayList<Neuron> outs = network.getOutputNeurons();

      float errorTotalDivOutOfThisNeuron=0;

      for (Neuron o : outs) {

        float errorOfOutputNeuronDivInputOfOutputNeuron = o.errorDivOut * o.outDivInput;
        float relatedW = o.connections.get(layerNeuronIndex).weight;

        float errorOfOutputNeuronDivOutOfThisNeuron = errorOfOutputNeuronDivInputOfOutputNeuron * relatedW;
        errorTotalDivOutOfThisNeuron += errorOfOutputNeuronDivOutOfThisNeuron;
      }

      outDivInput = this.outputValue * (1 - this.outputValue);

      for (Connection c : connections) {
        float inputDivW = c.from.outputValue;
        float errorTotalDivW = errorTotalDivOutOfThisNeuron * outDivInput * inputDivW;
        c.newWeight = c.newWeight - learningRate * errorTotalDivW;
      }
    }
  }

  void display() {
    stroke(0);
    strokeWeight(1);
    fill(150);
    ellipse(position.x, position.y, 30, 30);
  }
}