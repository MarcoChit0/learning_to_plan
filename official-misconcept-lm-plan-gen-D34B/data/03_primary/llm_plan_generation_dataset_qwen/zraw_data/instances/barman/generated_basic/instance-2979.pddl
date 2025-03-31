(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot261 shot308 - shot
      ingredient490 ingredient367 ingredient26 ingredient78 - ingredient
      cocktail411 - cocktail
      dispenser457 dispenser61 dispenser378 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot261)
  (ontable shot308)
  (dispenses dispenser457 ingredient490)
  (dispenses dispenser61 ingredient367)
  (dispenses dispenser378 ingredient26)
  (dispenses dispenser402 ingredient78)
  (clean shaker49)
  (clean shot261)
  (clean shot308)
  (empty shaker49)
  (empty shot261)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail411 ingredient490)
  (cocktail-part2 cocktail411 ingredient367)
)
 (:goal
  (and
      (contains shot261 cocktail411)
)))
