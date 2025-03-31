(define (problem prob)
 (:domain barman)
 (:objects 
      shaker473 - shaker
      left right - hand
      shot476 shot130 - shot
      ingredient432 ingredient32 - ingredient
      cocktail457 - cocktail
      dispenser393 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker473)
  (ontable shot476)
  (ontable shot130)
  (dispenses dispenser393 ingredient432)
  (dispenses dispenser273 ingredient32)
  (clean shaker473)
  (clean shot476)
  (clean shot130)
  (empty shaker473)
  (empty shot476)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker473 l0)
  (shaker-level shaker473 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient432)
  (cocktail-part2 cocktail457 ingredient32)
)
 (:goal
  (and
      (contains shot476 cocktail457)
)))
