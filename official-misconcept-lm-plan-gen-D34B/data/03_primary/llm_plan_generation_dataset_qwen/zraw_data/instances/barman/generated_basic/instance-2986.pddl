(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot457 shot248 - shot
      ingredient436 ingredient71 ingredient74 ingredient451 - ingredient
      cocktail13 - cocktail
      dispenser331 dispenser496 dispenser345 dispenser160 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot457)
  (ontable shot248)
  (dispenses dispenser331 ingredient436)
  (dispenses dispenser496 ingredient71)
  (dispenses dispenser345 ingredient74)
  (dispenses dispenser160 ingredient451)
  (clean shaker386)
  (clean shot457)
  (clean shot248)
  (empty shaker386)
  (empty shot457)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient436)
  (cocktail-part2 cocktail13 ingredient71)
)
 (:goal
  (and
      (contains shot457 cocktail13)
)))
