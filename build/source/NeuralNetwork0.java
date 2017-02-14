import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class NeuralNetwork0 extends PApplet {

Network network;
int layers = 2;
int neuronCountPerLayer = 2;
int outputLayerCount = 2;
int inputLayerCount = 2;
float learningRate = 0.05f;
int counter=0;
boolean start=false;

public void setup() {
  
  //randomSeed(3);

  network = new Network(width/2, height/2);

  Neuron output0 = new Neuron(false, 0, true, 0.05f, 0, 250, -75);
  Neuron output1 = new Neuron(false, 0, true, 0.95f, 1, 250, 75);

  for (int layer = 0; layer < layers; layer++) {
    for (int j = 0; j < neuronCountPerLayer; j++) {

      float x = map(layer, 0, layers, -250, 300);
      float y = map(j, 0, neuronCountPerLayer-1, -75, 75);

      Neuron n;
      if (layer == 0) {
        n = new Neuron(true, random(1), false, 0, j, x, y);
      } else {
        n = new Neuron(j, x, y);
      }

      //if hidden layer
      if (layer > 0) {
        for (int k = 0; k < neuronCountPerLayer; k++) {
          Neuron prev = network.neurons.get(network.neurons.size()-neuronCountPerLayer+k-j);
          float w = random(1);
          network.connect(prev, n, w);
        }
      }

      //if last hidden layer then connect with output neurons
      if (layer == layers-1) {
        network.connect(n, output0, random(1));
        network.connect(n, output1, random(1));
      }
      network.addNeuron(n);
    }
  }

  //add output neurons to network
  network.addNeuron(output0);
  network.addNeuron(output1);

  frameRate(120);

  network.feedforward();
}

public void mousePressed() {
  if (!start) {
    start=true;
  } else {
    Neuron n =network.neurons.get(network.neurons.size()-1);
    n.targetValue=random(1);

    Neuron n1 =network.neurons.get(network.neurons.size()-2);
    n1.targetValue=random(1);
  }
}

public void draw() {
  background(255);
  float x = 10;
  float y = 50;

  network.display();

  fill(0);
  text("iteration: "+counter, 10, 30);
  int i=0;
  for (Neuron n : network.neurons) {

    text("o: "+n.outputValue, x, y);
    y+=20;
  }

  //text("hedef: 0.05",100,y-40);
  //text("hedef: 0.95",100,y-20);

  if (start) {
    network.feedforward();
    counter++;
  }
}
class Connection {
  Neuron from;
  Neuron to;
  float weight;
  float newWeight;
  float output = 0;
  int col;
  Connection(Neuron from, Neuron to, float w) {
    weight = w;
    newWeight = w;
    this.from = from;
    this.to = to;
    col = color(random(255),random(255),random(255));
  }

  public void display() {
    stroke(col);
    strokeWeight(max(1,1+weight*20));
    line(from.position.x, from.position.y, to.position.x, to.position.y);
    
  }
}
class Network {
  ArrayList<Neuron> neurons;
  ArrayList<Connection> connections;
  int activeLayer = 1;
  float[] bias;
  float error;
  boolean isBackwardPassing=false;
  PVector position;

  Network(float x, float y) {
    position = new PVector(x, y);
    neurons = new ArrayList<Neuron>();
    connections = new ArrayList<Connection>();
    bias = new float[layers];
    for (int i=0; i< layers; i++) {
      bias[i] = random(1);
    }
  }

  public void addNeuron(Neuron n) {
    neurons.add(n);
  }

  public void connect(Neuron prev, Neuron newNeuron, float weight) {
    Connection c = new Connection(prev, newNeuron, weight);
    newNeuron.addConnection(c);
    connections.add(c);
  } 

  public void feedforward() {

    if (!isBackwardPassing) {
      for (int i=activeLayer * neuronCountPerLayer; i<(activeLayer * neuronCountPerLayer) + neuronCountPerLayer; i++) {
        Neuron n = neurons.get(i);
        n.calculate();
      }

      activeLayer++;

      if (activeLayer<=layers) {
        feedforward();
      } else {
        isBackwardPassing = true;
        calculateOutputErrors();
        backPass();
      }
    }
  }

  public void backPass() {
    if (isBackwardPassing) {
      for (int i=neurons.size()-1; i>= inputLayerCount; i--) {
        Neuron n = neurons.get( i );
        n.backwardPass();
      }
      for (Connection c : connections) {
        c.weight = c.newWeight;
      }
      isBackwardPassing=false;
      activeLayer=1;
      //printOut();
    }
  }

  public void printOut() {
    int i=0;
    for (Neuron n : neurons) {
      println("n" + i + ": " + n.outputValue);
      i++;
    }
    
    println("----------");
    
    i=0;
    for (Connection c : connections) {
      println("c.w" + i + ": " + c.weight);
      i++;
    }
    
    println("----------");
  }

  public void calculateOutputErrors() {
    float errorTotal=0;
    for (int i=0; i< outputLayerCount; i++) {
      Neuron n = neurons.get( (neurons.size()-1)-i );
      float diff = n.targetValue - n.outputValue;
      n.error = 0.5f * (diff * diff);
      errorTotal += n.error;
      //println("output neuron error:" + n.error);
    }
    error = errorTotal;

    //println("total error:" + error);
  }

  public ArrayList<Neuron> getOutputNeurons() {
    ArrayList<Neuron> outputs = new ArrayList<Neuron>();
    for (int i=0; i< outputLayerCount; i++) {
      Neuron n = neurons.get( (neurons.size()-1)-i );
      outputs.add(n);
    }
    return outputs;
  }

  public void display() {
    pushMatrix();
    translate(position.x, position.y);
    for (Neuron n : neurons) {
      n.display();
    }

    for (Connection c : connections) {
      c.display();
    }
    popMatrix();
  }
}
class Neuron {
  PVector position;
  int layerIndex;

  ArrayList<Connection> connections;

  boolean isOutput;
  float targetValue;

  boolean isInput;

  float outputValue;
  float error;

  float errorDivOut;
  float outDivInput;

  //for input/output neuron
  Neuron(boolean isInput, float input, boolean isOutput, float target, int index, float x, float y) {
    position = new PVector(x, y);
    connections = new ArrayList<Connection>();
    this.isOutput = isOutput;
    this.targetValue = target;

    this.isInput = isInput;
    this.outputValue = input;

    this.layerIndex = index;
  }

  //for hidden neuron
  Neuron(int index, float x, float y) {
    position = new PVector(x, y);
    connections = new ArrayList<Connection>();
    this.isOutput = false;
    this.targetValue = 0;

    this.isInput = false;
    this.outputValue = 0;
    this.layerIndex = index;
  }

  public void addConnection(Connection c) {
    connections.add(c);
  }

  public void calculate() {
    float score =0;
    for (Connection c : connections) {
      score += c.weight * c.from.outputValue;
    }
    score = score + network.bias[ network.activeLayer-1 ];
    this.outputValue = 1/(1+ exp(score*-1));
  }

  public void backwardPass() {
    if (isOutput) {
      errorDivOut = (this.targetValue - this.outputValue) * -1;
      outDivInput = this.outputValue * (1 - this.outputValue);

      for (Connection c : connections) {
        float inputDivW = c.from.outputValue;
        float errorTotalDivW = errorDivOut * outDivInput * inputDivW;
        c.newWeight = c.newWeight - learningRate * errorTotalDivW;
      }
    } else if (!isOutput && !isInput) {

      ArrayList<Neuron> outs = network.getOutputNeurons();

      float errorTotalDivOutOfThisNeuron=0;

      for (Neuron o : outs) {

        float errorOfOutputNeuronDivInputOfOutputNeuron = o.errorDivOut * o.outDivInput;
        float relatedW = o.connections.get(layerIndex).weight;

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

  public void display() {
    stroke(0);
    strokeWeight(1);
    fill(150);
    ellipse(position.x, position.y, 30, 30);

  }
}
  public void settings() {  size(640, 360); }
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "NeuralNetwork0" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
