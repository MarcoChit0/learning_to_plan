(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot399 - shot
      ingredient53 ingredient303 ingredient366 - ingredient
      cocktail1 - cocktail
      dispenser100 dispenser402 dispenser302 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot399)
  (dispenses dispenser100 ingredient53)
  (dispenses dispenser402 ingredient303)
  (dispenses dispenser302 ingredient366)
  (clean shaker250)
  (clean shot399)
  (empty shaker250)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient366)
  (cocktail-part2 cocktail1 ingredient53)
)
 (:goal
  (and
      (contains shot399 cocktail1)
)))
