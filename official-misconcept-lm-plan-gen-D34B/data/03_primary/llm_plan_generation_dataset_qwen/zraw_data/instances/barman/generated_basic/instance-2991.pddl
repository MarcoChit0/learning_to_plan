(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot381 shot409 - shot
      ingredient81 ingredient274 ingredient408 ingredient279 - ingredient
      cocktail10 - cocktail
      dispenser141 dispenser302 dispenser405 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot381)
  (ontable shot409)
  (dispenses dispenser141 ingredient81)
  (dispenses dispenser302 ingredient274)
  (dispenses dispenser405 ingredient408)
  (dispenses dispenser390 ingredient279)
  (clean shaker62)
  (clean shot381)
  (clean shot409)
  (empty shaker62)
  (empty shot381)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient81)
  (cocktail-part2 cocktail10 ingredient279)
)
 (:goal
  (and
      (contains shot381 cocktail10)
)))
