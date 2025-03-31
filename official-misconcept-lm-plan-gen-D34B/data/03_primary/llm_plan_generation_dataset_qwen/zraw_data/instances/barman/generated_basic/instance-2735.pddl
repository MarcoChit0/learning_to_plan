(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot430 - shot
      ingredient233 ingredient262 - ingredient
      cocktail2 - cocktail
      dispenser92 dispenser329 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot430)
  (dispenses dispenser92 ingredient233)
  (dispenses dispenser329 ingredient262)
  (clean shaker351)
  (clean shot430)
  (empty shaker351)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient233)
  (cocktail-part2 cocktail2 ingredient262)
)
 (:goal
  (and
      (contains shot430 cocktail2)
)))
