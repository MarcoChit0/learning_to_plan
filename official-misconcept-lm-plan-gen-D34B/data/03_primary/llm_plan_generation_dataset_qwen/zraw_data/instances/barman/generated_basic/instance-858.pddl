(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot172 - shot
      ingredient127 ingredient212 ingredient282 - ingredient
      cocktail1 - cocktail
      dispenser232 dispenser172 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot172)
  (dispenses dispenser232 ingredient127)
  (dispenses dispenser172 ingredient212)
  (dispenses dispenser167 ingredient282)
  (clean shaker210)
  (clean shot172)
  (empty shaker210)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient282)
  (cocktail-part2 cocktail1 ingredient212)
)
 (:goal
  (and
      (contains shot172 cocktail1)
)))
