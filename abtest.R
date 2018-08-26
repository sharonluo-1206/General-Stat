
## A/B test on two-sample proportion 

### return the sample size required for each arm in an 50/50 split A/B test
size_cal <- function(p1, uplift = 0.2, alpha = 0.05, beta = 0.2, twosided = TRUE) {
  p2 = (1+uplift)*p1;
  if(twosided == TRUE) {
    s = (qnorm(alpha/2)+qnorm(beta))^2*(p1*(1-p1)+p2*(1-p2))/(p1-p2)^2
  }else{
    s = (qnorm(alpha)+qnorm(beta))^2*(p1*(1-p1)+p2*(1-p2))/(p1-p2)^2
  }
  s
}


### return minimum percentage uplift required to be claimed as statistically significant in an 50/50 split A/B test
uplift <- function(p1, size, alpha = 0.05, beta = 0.2, twosided =  TRUE) {
  if (twosided == TRUE) {
    zvalue = qnorm(alpha/2)
  }else{
    zvalue = qnorm(alpha)
  }
  a = size + (zvalue + qnorm(beta))^2
  b = -2*size*p1 - (zvalue + qnorm(beta))^2
  c = -(zvalue + qnorm(beta))^2*p1*(1-p1)+size*p1^2
  p2 = (-b + sqrt(b^2-4*a*c))/(2*a)
  p2/p1-1
}