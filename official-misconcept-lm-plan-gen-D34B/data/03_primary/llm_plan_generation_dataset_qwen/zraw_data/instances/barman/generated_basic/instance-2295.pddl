(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot171 - shot
      ingredient282 ingredient49 ingredient102 - ingredient
      cocktail124 - cocktail
      dispenser435 dispenser39 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot171)
  (dispenses dispenser435 ingredient282)
  (dispenses dispenser39 ingredient49)
  (dispenses dispenser13 ingredient102)
  (clean shaker324)
  (clean shot171)
  (empty shaker324)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient102)
  (cocktail-part2 cocktail124 ingredient282)
)
 (:goal
  (and
      (contains shot171 cocktail124)
)))
