class Canvas{
  
  float marginLeft=0;
  float marginRight=0;
  float marginTop=0;
  float marginBottom=0;
  
  float padding=50;

  float x=0;
  float y=0;
  float w=0;
  float h=0;
  
  color bgCol;
  
  Canvas(float x, float y, float w, float h, float margin, color col){
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    
    this.marginLeft = margin;
    this.marginRight = margin;
    this.marginTop = margin;
    this.marginBottom = margin;
    bgCol = col;
  }
  
  void show(){
    noStroke();
    fill(bgCol);
    rect(x+marginLeft,y+marginRight,w-marginLeft-marginRight,h-marginTop-marginBottom);
  }
}