(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot67 - shot
      ingredient102 ingredient120 - ingredient
      cocktail83 - cocktail
      dispenser236 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot67)
  (dispenses dispenser236 ingredient102)
  (dispenses dispenser143 ingredient120)
  (clean shaker95)
  (clean shot67)
  (empty shaker95)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient120)
  (cocktail-part2 cocktail83 ingredient102)
)
 (:goal
  (and
      (contains shot67 cocktail83)
)))
