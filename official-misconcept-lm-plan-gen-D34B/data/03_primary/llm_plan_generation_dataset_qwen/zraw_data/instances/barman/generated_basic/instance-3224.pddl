(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot399 - shot
      ingredient106 ingredient301 - ingredient
      cocktail379 - cocktail
      dispenser458 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot399)
  (dispenses dispenser458 ingredient106)
  (dispenses dispenser69 ingredient301)
  (clean shaker382)
  (clean shot399)
  (empty shaker382)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail379 ingredient301)
  (cocktail-part2 cocktail379 ingredient106)
)
 (:goal
  (and
      (contains shot399 cocktail379)
)))
