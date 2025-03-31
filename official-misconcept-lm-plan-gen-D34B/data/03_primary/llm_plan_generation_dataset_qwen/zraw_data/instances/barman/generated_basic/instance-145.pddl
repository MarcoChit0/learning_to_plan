(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot308 - shot
      ingredient437 ingredient15 ingredient256 ingredient195 - ingredient
      cocktail1 - cocktail
      dispenser334 dispenser162 dispenser140 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot308)
  (dispenses dispenser334 ingredient437)
  (dispenses dispenser162 ingredient15)
  (dispenses dispenser140 ingredient256)
  (dispenses dispenser492 ingredient195)
  (clean shaker294)
  (clean shot308)
  (empty shaker294)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient195)
  (cocktail-part2 cocktail1 ingredient256)
)
 (:goal
  (and
      (contains shot308 cocktail1)
)))
