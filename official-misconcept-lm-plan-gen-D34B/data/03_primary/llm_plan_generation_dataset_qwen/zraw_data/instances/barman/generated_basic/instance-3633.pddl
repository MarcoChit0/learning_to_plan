(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot405 - shot
      ingredient210 ingredient166 ingredient487 ingredient454 - ingredient
      cocktail480 - cocktail
      dispenser3 dispenser272 dispenser153 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot405)
  (dispenses dispenser3 ingredient210)
  (dispenses dispenser272 ingredient166)
  (dispenses dispenser153 ingredient487)
  (dispenses dispenser421 ingredient454)
  (clean shaker34)
  (clean shot405)
  (empty shaker34)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient210)
  (cocktail-part2 cocktail480 ingredient454)
)
 (:goal
  (and
      (contains shot405 cocktail480)
)))
