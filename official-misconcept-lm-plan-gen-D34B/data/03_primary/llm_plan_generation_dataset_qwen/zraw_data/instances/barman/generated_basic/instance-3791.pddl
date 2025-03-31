(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot64 - shot
      ingredient94 ingredient63 ingredient284 ingredient5 - ingredient
      cocktail189 - cocktail
      dispenser372 dispenser376 dispenser279 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot64)
  (dispenses dispenser372 ingredient94)
  (dispenses dispenser376 ingredient63)
  (dispenses dispenser279 ingredient284)
  (dispenses dispenser285 ingredient5)
  (clean shaker222)
  (clean shot64)
  (empty shaker222)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail189 ingredient5)
  (cocktail-part2 cocktail189 ingredient284)
)
 (:goal
  (and
      (contains shot64 cocktail189)
)))
