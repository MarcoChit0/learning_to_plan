(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot474 shot375 shot283 - shot
      ingredient67 ingredient395 ingredient208 ingredient413 - ingredient
      cocktail245 - cocktail
      dispenser283 dispenser442 dispenser420 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot474)
  (ontable shot375)
  (ontable shot283)
  (dispenses dispenser283 ingredient67)
  (dispenses dispenser442 ingredient395)
  (dispenses dispenser420 ingredient208)
  (dispenses dispenser463 ingredient413)
  (clean shaker397)
  (clean shot474)
  (clean shot375)
  (clean shot283)
  (empty shaker397)
  (empty shot474)
  (empty shot375)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient395)
  (cocktail-part2 cocktail245 ingredient413)
)
 (:goal
  (and
      (contains shot474 cocktail245)
      (contains shot375 cocktail245)
)))
