(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot226 shot234 shot71 - shot
      ingredient182 ingredient419 ingredient46 ingredient94 - ingredient
      cocktail333 - cocktail
      dispenser362 dispenser351 dispenser62 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot226)
  (ontable shot234)
  (ontable shot71)
  (dispenses dispenser362 ingredient182)
  (dispenses dispenser351 ingredient419)
  (dispenses dispenser62 ingredient46)
  (dispenses dispenser60 ingredient94)
  (clean shaker238)
  (clean shot226)
  (clean shot234)
  (clean shot71)
  (empty shaker238)
  (empty shot226)
  (empty shot234)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail333 ingredient419)
  (cocktail-part2 cocktail333 ingredient94)
)
 (:goal
  (and
      (contains shot226 cocktail333)
      (contains shot234 ingredient419)
)))
