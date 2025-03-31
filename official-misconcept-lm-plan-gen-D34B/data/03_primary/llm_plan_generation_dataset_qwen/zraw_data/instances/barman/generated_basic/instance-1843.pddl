(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot394 shot182 - shot
      ingredient63 ingredient105 ingredient298 - ingredient
      cocktail476 - cocktail
      dispenser406 dispenser293 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot394)
  (ontable shot182)
  (dispenses dispenser406 ingredient63)
  (dispenses dispenser293 ingredient105)
  (dispenses dispenser260 ingredient298)
  (clean shaker408)
  (clean shot394)
  (clean shot182)
  (empty shaker408)
  (empty shot394)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient105)
  (cocktail-part2 cocktail476 ingredient63)
)
 (:goal
  (and
      (contains shot394 cocktail476)
)))
