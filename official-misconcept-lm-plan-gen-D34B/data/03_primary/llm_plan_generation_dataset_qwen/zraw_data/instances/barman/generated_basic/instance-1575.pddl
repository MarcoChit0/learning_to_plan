(define (problem prob)
 (:domain barman)
 (:objects 
      shaker175 - shaker
      left right - hand
      shot340 shot477 shot493 - shot
      ingredient73 ingredient121 ingredient279 ingredient380 - ingredient
      cocktail166 - cocktail
      dispenser132 dispenser155 dispenser24 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker175)
  (ontable shot340)
  (ontable shot477)
  (ontable shot493)
  (dispenses dispenser132 ingredient73)
  (dispenses dispenser155 ingredient121)
  (dispenses dispenser24 ingredient279)
  (dispenses dispenser208 ingredient380)
  (clean shaker175)
  (clean shot340)
  (clean shot477)
  (clean shot493)
  (empty shaker175)
  (empty shot340)
  (empty shot477)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker175 l0)
  (shaker-level shaker175 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail166 ingredient380)
  (cocktail-part2 cocktail166 ingredient279)
)
 (:goal
  (and
      (contains shot340 cocktail166)
      (contains shot477 cocktail166)
)))
