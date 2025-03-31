(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot344 shot39 - shot
      ingredient349 ingredient478 ingredient211 ingredient104 - ingredient
      cocktail218 - cocktail
      dispenser333 dispenser36 dispenser57 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot344)
  (ontable shot39)
  (dispenses dispenser333 ingredient349)
  (dispenses dispenser36 ingredient478)
  (dispenses dispenser57 ingredient211)
  (dispenses dispenser427 ingredient104)
  (clean shaker391)
  (clean shot344)
  (clean shot39)
  (empty shaker391)
  (empty shot344)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail218 ingredient211)
  (cocktail-part2 cocktail218 ingredient349)
)
 (:goal
  (and
      (contains shot344 cocktail218)
)))
