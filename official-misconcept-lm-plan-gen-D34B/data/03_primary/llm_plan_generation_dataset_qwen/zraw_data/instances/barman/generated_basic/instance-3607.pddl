(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot202 shot334 shot479 - shot
      ingredient171 ingredient305 ingredient447 - ingredient
      cocktail275 - cocktail
      dispenser443 dispenser201 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot202)
  (ontable shot334)
  (ontable shot479)
  (dispenses dispenser443 ingredient171)
  (dispenses dispenser201 ingredient305)
  (dispenses dispenser398 ingredient447)
  (clean shaker233)
  (clean shot202)
  (clean shot334)
  (clean shot479)
  (empty shaker233)
  (empty shot202)
  (empty shot334)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail275 ingredient171)
  (cocktail-part2 cocktail275 ingredient305)
)
 (:goal
  (and
      (contains shot202 cocktail275)
      (contains shot334 cocktail275)
)))
