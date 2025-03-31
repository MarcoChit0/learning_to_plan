(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot64 - shot
      ingredient397 ingredient72 - ingredient
      cocktail1 - cocktail
      dispenser36 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot64)
  (dispenses dispenser36 ingredient397)
  (dispenses dispenser447 ingredient72)
  (clean shaker282)
  (clean shot64)
  (empty shaker282)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient72)
)
 (:goal
  (and
      (contains shot64 cocktail1)
)))
