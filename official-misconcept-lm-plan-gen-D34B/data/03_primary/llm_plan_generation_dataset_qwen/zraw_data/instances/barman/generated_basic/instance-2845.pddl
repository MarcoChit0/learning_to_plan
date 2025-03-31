(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot453 shot392 - shot
      ingredient363 ingredient13 ingredient198 ingredient290 - ingredient
      cocktail79 - cocktail
      dispenser100 dispenser493 dispenser378 dispenser435 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot453)
  (ontable shot392)
  (dispenses dispenser100 ingredient363)
  (dispenses dispenser493 ingredient13)
  (dispenses dispenser378 ingredient198)
  (dispenses dispenser435 ingredient290)
  (clean shaker390)
  (clean shot453)
  (clean shot392)
  (empty shaker390)
  (empty shot453)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail79 ingredient290)
  (cocktail-part2 cocktail79 ingredient13)
)
 (:goal
  (and
      (contains shot453 cocktail79)
)))
