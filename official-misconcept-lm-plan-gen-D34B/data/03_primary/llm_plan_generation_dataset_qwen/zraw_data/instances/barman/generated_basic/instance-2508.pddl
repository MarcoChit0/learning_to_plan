(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot399 - shot
      ingredient154 ingredient239 - ingredient
      cocktail37 - cocktail
      dispenser130 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot399)
  (dispenses dispenser130 ingredient154)
  (dispenses dispenser373 ingredient239)
  (clean shaker282)
  (clean shot399)
  (empty shaker282)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail37 ingredient154)
  (cocktail-part2 cocktail37 ingredient239)
)
 (:goal
  (and
      (contains shot399 cocktail37)
)))
