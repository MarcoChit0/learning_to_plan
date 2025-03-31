(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot196 - shot
      ingredient493 ingredient188 ingredient62 - ingredient
      cocktail1 - cocktail
      dispenser396 dispenser143 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot196)
  (dispenses dispenser396 ingredient493)
  (dispenses dispenser143 ingredient188)
  (dispenses dispenser261 ingredient62)
  (clean shaker499)
  (clean shot196)
  (empty shaker499)
  (empty shot196)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient493)
  (cocktail-part2 cocktail1 ingredient188)
)
 (:goal
  (and
      (contains shot196 cocktail1)
)))
