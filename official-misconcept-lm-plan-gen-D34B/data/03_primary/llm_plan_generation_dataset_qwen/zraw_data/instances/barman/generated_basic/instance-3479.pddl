(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot311 shot139 shot83 - shot
      ingredient300 ingredient252 ingredient454 - ingredient
      cocktail71 - cocktail
      dispenser290 dispenser94 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot311)
  (ontable shot139)
  (ontable shot83)
  (dispenses dispenser290 ingredient300)
  (dispenses dispenser94 ingredient252)
  (dispenses dispenser191 ingredient454)
  (clean shaker72)
  (clean shot311)
  (clean shot139)
  (clean shot83)
  (empty shaker72)
  (empty shot311)
  (empty shot139)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail71 ingredient300)
  (cocktail-part2 cocktail71 ingredient454)
)
 (:goal
  (and
      (contains shot311 cocktail71)
      (contains shot139 cocktail71)
)))
