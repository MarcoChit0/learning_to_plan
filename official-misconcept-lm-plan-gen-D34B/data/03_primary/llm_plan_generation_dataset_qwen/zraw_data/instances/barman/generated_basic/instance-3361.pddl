(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot372 - shot
      ingredient166 ingredient201 - ingredient
      cocktail476 - cocktail
      dispenser283 dispenser228 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot372)
  (dispenses dispenser283 ingredient166)
  (dispenses dispenser228 ingredient201)
  (clean shaker352)
  (clean shot372)
  (empty shaker352)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient201)
  (cocktail-part2 cocktail476 ingredient166)
)
 (:goal
  (and
      (contains shot372 cocktail476)
)))
