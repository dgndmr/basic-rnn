class Connection {
  Neuron from;
  Neuron to;
  float weight;
  float newWeight;
  float output = 0;
  color col;
  
  Connection(Neuron from, Neuron to, float w) {
    weight = w;
    newWeight = w;  //after back propogation done => newWeight assignes to weight
    this.from = from;
    this.to = to;
    col = color(random(255), random(255), random(255));
  }

  void display() {
    stroke(col, 100);
    strokeWeight(1+abs(weight*2) );
    line(from.position.x, from.position.y, to.position.x, to.position.y);
  }
}