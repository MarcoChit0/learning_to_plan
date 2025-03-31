(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot321 shot120 shot397 - shot
      ingredient373 ingredient385 - ingredient
      cocktail262 - cocktail
      dispenser499 dispenser401 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot321)
  (ontable shot120)
  (ontable shot397)
  (dispenses dispenser499 ingredient373)
  (dispenses dispenser401 ingredient385)
  (clean shaker363)
  (clean shot321)
  (clean shot120)
  (clean shot397)
  (empty shaker363)
  (empty shot321)
  (empty shot120)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient373)
  (cocktail-part2 cocktail262 ingredient385)
)
 (:goal
  (and
      (contains shot321 cocktail262)
      (contains shot120 ingredient373)
)))
