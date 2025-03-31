(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot207 - shot
      ingredient3 ingredient19 - ingredient
      cocktail137 - cocktail
      dispenser160 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot207)
  (dispenses dispenser160 ingredient3)
  (dispenses dispenser88 ingredient19)
  (clean shaker273)
  (clean shot207)
  (empty shaker273)
  (empty shot207)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail137 ingredient3)
  (cocktail-part2 cocktail137 ingredient19)
)
 (:goal
  (and
      (contains shot207 cocktail137)
)))
