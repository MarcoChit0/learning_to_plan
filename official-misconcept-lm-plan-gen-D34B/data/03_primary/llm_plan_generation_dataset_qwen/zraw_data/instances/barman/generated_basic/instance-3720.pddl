(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot339 - shot
      ingredient4 ingredient64 ingredient420 ingredient24 - ingredient
      cocktail476 - cocktail
      dispenser122 dispenser423 dispenser472 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot339)
  (dispenses dispenser122 ingredient4)
  (dispenses dispenser423 ingredient64)
  (dispenses dispenser472 ingredient420)
  (dispenses dispenser114 ingredient24)
  (clean shaker294)
  (clean shot339)
  (empty shaker294)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient420)
  (cocktail-part2 cocktail476 ingredient4)
)
 (:goal
  (and
      (contains shot339 cocktail476)
)))
