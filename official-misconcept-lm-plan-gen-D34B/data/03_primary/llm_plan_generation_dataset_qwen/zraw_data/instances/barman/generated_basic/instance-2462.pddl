(define (problem prob)
 (:domain barman)
 (:objects 
      shaker14 - shaker
      left right - hand
      shot370 shot403 shot334 - shot
      ingredient44 ingredient403 ingredient115 ingredient65 - ingredient
      cocktail454 - cocktail
      dispenser81 dispenser184 dispenser359 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker14)
  (ontable shot370)
  (ontable shot403)
  (ontable shot334)
  (dispenses dispenser81 ingredient44)
  (dispenses dispenser184 ingredient403)
  (dispenses dispenser359 ingredient115)
  (dispenses dispenser412 ingredient65)
  (clean shaker14)
  (clean shot370)
  (clean shot403)
  (clean shot334)
  (empty shaker14)
  (empty shot370)
  (empty shot403)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker14 l0)
  (shaker-level shaker14 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail454 ingredient44)
  (cocktail-part2 cocktail454 ingredient65)
)
 (:goal
  (and
      (contains shot370 cocktail454)
      (contains shot403 cocktail454)
)))
