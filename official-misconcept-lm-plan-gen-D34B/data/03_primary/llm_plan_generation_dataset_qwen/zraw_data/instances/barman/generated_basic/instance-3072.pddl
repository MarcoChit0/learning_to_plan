(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot306 shot208 - shot
      ingredient452 ingredient206 - ingredient
      cocktail438 - cocktail
      dispenser323 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot306)
  (ontable shot208)
  (dispenses dispenser323 ingredient452)
  (dispenses dispenser208 ingredient206)
  (clean shaker416)
  (clean shot306)
  (clean shot208)
  (empty shaker416)
  (empty shot306)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail438 ingredient206)
  (cocktail-part2 cocktail438 ingredient452)
)
 (:goal
  (and
      (contains shot306 cocktail438)
)))
