#let dv = $dot(v)$
#let bv = $bold(v)$
#let IL = $I_"L"$
#let IK = $I_"K"$
#let INa = $I_"Na"$
#let IT = $I_"T"$
#let ICa = $I_"Ca"$
#let IAHP = $I_"AHP"$
#let IGS = $I_("G" -> "S")$
#let ISG = $I_("S" -> "G")$
#let IGG = $I_("G" -> "G")$
#let par(x) = text(fill: olive, $#x$)
#let pr(x) = $#x _"pre" med$
#let po(x) = $#x _"post" med$
#set page(number-align: center, numbering: "I")
#show heading.where(level: 2): set heading(numbering: "1.", level: 1)
//#show math.equation: set text(size: 10pt)

#align(center)[= CBGT model equations]

== STN 
$
par(C_"m") bold(dv) &= - IL - IK - INa - IT - ICa - IAHP - IGS\
IL &= par(g_"L") (bv - par(v_"L"))\
IK &= par(g_"K") bold(n)^4 (bv - par(v_"K")) \
INa &= par(g_"Na") m^3_oo bold(h) (bv - par(v_"Na")) \
IT &= par(g_"T") a^3_oo b^2_oo (bv - par(v_"Ca")) \
ICa &= par(g_"Ca") s^2_oo (bv - par(v_"Ca")) \
b_oo  &= (1 + exp(frac(bold(r) - par(theta_b), par(sigma_b)))^(-1)) 
          - (1 + exp(-frac(par(theta_b), par(sigma_b))))^(-1)\
x_oo  &= (1 + exp(-frac(bv - par(theta_x), par(sigma_x))))^(-1) &, x &in {n, m, h, a, r, s} \
lr(\{med #block($
  bold(dot(x)) &= par(Phi_x) frac(x_oo - x, tau_x) \
  tau_x &= par(tau^0_x) + par(tau^1_x) (1 + exp(-frac(v - par(theta^tau_x), par(sigma^tau_x))))
$)) #h(-14.2em) && ,x &in {n, h, r} \
IAHP &= par(g_"AHP") (v - par(v_"K")) frac(["Ca"], ["Ca"] + par(k_1)) \
dot(["Ca"]) &= par(epsilon) (-ICa - IT - par(k_"Ca")["Ca"]) \
dot(s) &= par(alpha) H_oo (v - par(theta_g))(1 - s) - par(beta)s \
H_oo (v) &= (1 + exp(-frac(v - par(theta^H_g), par(sigma^H_g))))^(-1) \
IGS &=  par(g_("G" -> "S")) (bv - par(v_("G" -> "S"))) sum_(j in"GPe") w_j bold(s_j)\
$ 

== GPe
Same as STN except

$
par(C_"m") bold(dv) &= - IL - IK - INa - IT - ICa - IAHP - ISG - IGG + par(I_"app") \
IT &= par(g_"T") a^3_oo bold(r) (bv - par(v_"Ca")) \
bold(dot(r)) &= par(Phi_r) frac(r_oo - bv, par(tau_r))
$

== STDP

$
 pr(dot(p)) &= -frac(pr(p), par(pr(tau))) + delta(t - t_"spike") \
 po(dot(p)) &= -frac(po(p), par(po(tau))) + delta(t - t_"spike") \
 "Where" quad delta(t) &=  cases(
   0 &quad "if" x != 0,
   oo &quad "if" x = 0)quad, "s.t." & integral_(-oo)^oo delta(t) d t = 1
$ 
#v(100%)

For a weight from neuron $i in I$ to  neuron $j in J$
$
 Delta w_(i,j) &= cases(
   &par(pr(A^J)) bold(pr(p^i)) &quad "if" t = t^i_"spike",
   -&par(po(A^J)) bold(po(p^j)) &quad "if" t = t^i_"spike",
   &0 &quad "otherwise"
 )
$
Or in matrix form
$
Delta W &= eta lr((par(pr(A^J)) ((bold(p^I_"pre"))^top dot bb(1)_{bold(po(p^J)) = 1}) 
                  -par(po(A^J)) ((bold(p^J_"post"))^top dot bb(1)_{bold(po(p^I)) = 1})), size: #110%) 
           dot.circle par(C_(I->J)) \
$

#v(1fr)

#align(right)[
= Legend
\ 
$par(x)  -> "Parameters" \ 
 bold(x) -> "State Variables" $]
