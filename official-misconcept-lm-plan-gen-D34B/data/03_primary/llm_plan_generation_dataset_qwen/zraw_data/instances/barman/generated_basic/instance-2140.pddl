(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot153 - shot
      ingredient471 ingredient441 - ingredient
      cocktail433 - cocktail
      dispenser465 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot153)
  (dispenses dispenser465 ingredient471)
  (dispenses dispenser233 ingredient441)
  (clean shaker224)
  (clean shot153)
  (empty shaker224)
  (empty shot153)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail433 ingredient471)
  (cocktail-part2 cocktail433 ingredient441)
)
 (:goal
  (and
      (contains shot153 cocktail433)
)))
