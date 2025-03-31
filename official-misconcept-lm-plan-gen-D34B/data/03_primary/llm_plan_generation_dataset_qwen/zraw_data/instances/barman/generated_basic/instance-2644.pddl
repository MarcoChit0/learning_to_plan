(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot296 shot234 - shot
      ingredient371 ingredient74 ingredient279 - ingredient
      cocktail238 - cocktail
      dispenser44 dispenser480 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot296)
  (ontable shot234)
  (dispenses dispenser44 ingredient371)
  (dispenses dispenser480 ingredient74)
  (dispenses dispenser430 ingredient279)
  (clean shaker474)
  (clean shot296)
  (clean shot234)
  (empty shaker474)
  (empty shot296)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail238 ingredient74)
  (cocktail-part2 cocktail238 ingredient279)
)
 (:goal
  (and
      (contains shot296 cocktail238)
)))
