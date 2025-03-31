(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot230 - shot
      ingredient85 ingredient384 ingredient381 ingredient490 - ingredient
      cocktail1 - cocktail
      dispenser423 dispenser148 dispenser184 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot230)
  (dispenses dispenser423 ingredient85)
  (dispenses dispenser148 ingredient384)
  (dispenses dispenser184 ingredient381)
  (dispenses dispenser381 ingredient490)
  (clean shaker6)
  (clean shot230)
  (empty shaker6)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient85)
  (cocktail-part2 cocktail1 ingredient384)
)
 (:goal
  (and
      (contains shot230 cocktail1)
)))
