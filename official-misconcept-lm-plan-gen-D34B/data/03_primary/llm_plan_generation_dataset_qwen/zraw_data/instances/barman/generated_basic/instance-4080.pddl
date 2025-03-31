(define (problem prob)
 (:domain barman)
 (:objects 
      shaker245 - shaker
      left right - hand
      shot32 shot108 - shot
      ingredient125 ingredient466 ingredient19 ingredient376 - ingredient
      cocktail367 - cocktail
      dispenser357 dispenser51 dispenser490 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker245)
  (ontable shot32)
  (ontable shot108)
  (dispenses dispenser357 ingredient125)
  (dispenses dispenser51 ingredient466)
  (dispenses dispenser490 ingredient19)
  (dispenses dispenser167 ingredient376)
  (clean shaker245)
  (clean shot32)
  (clean shot108)
  (empty shaker245)
  (empty shot32)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker245 l0)
  (shaker-level shaker245 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient19)
  (cocktail-part2 cocktail367 ingredient376)
)
 (:goal
  (and
      (contains shot32 cocktail367)
)))
