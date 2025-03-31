(define (problem prob)
 (:domain barman)
 (:objects 
      shaker346 - shaker
      left right - hand
      shot497 - shot
      ingredient80 ingredient107 ingredient285 ingredient36 - ingredient
      cocktail220 - cocktail
      dispenser112 dispenser49 dispenser420 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker346)
  (ontable shot497)
  (dispenses dispenser112 ingredient80)
  (dispenses dispenser49 ingredient107)
  (dispenses dispenser420 ingredient285)
  (dispenses dispenser431 ingredient36)
  (clean shaker346)
  (clean shot497)
  (empty shaker346)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker346 l0)
  (shaker-level shaker346 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient285)
  (cocktail-part2 cocktail220 ingredient107)
)
 (:goal
  (and
      (contains shot497 cocktail220)
)))
