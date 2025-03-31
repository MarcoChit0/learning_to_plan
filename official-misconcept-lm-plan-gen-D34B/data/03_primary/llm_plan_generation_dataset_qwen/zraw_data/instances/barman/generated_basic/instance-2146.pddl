(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot94 - shot
      ingredient240 ingredient53 - ingredient
      cocktail346 - cocktail
      dispenser268 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot94)
  (dispenses dispenser268 ingredient240)
  (dispenses dispenser146 ingredient53)
  (clean shaker468)
  (clean shot94)
  (empty shaker468)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient53)
  (cocktail-part2 cocktail346 ingredient240)
)
 (:goal
  (and
      (contains shot94 cocktail346)
)))
