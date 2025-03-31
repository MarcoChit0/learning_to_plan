(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot399 - shot
      ingredient130 ingredient169 ingredient423 ingredient96 - ingredient
      cocktail481 - cocktail
      dispenser19 dispenser336 dispenser150 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot399)
  (dispenses dispenser19 ingredient130)
  (dispenses dispenser336 ingredient169)
  (dispenses dispenser150 ingredient423)
  (dispenses dispenser257 ingredient96)
  (clean shaker266)
  (clean shot399)
  (empty shaker266)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail481 ingredient130)
  (cocktail-part2 cocktail481 ingredient96)
)
 (:goal
  (and
      (contains shot399 cocktail481)
)))
