(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot131 shot215 shot60 - shot
      ingredient129 ingredient401 - ingredient
      cocktail352 - cocktail
      dispenser24 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot131)
  (ontable shot215)
  (ontable shot60)
  (dispenses dispenser24 ingredient129)
  (dispenses dispenser57 ingredient401)
  (clean shaker75)
  (clean shot131)
  (clean shot215)
  (clean shot60)
  (empty shaker75)
  (empty shot131)
  (empty shot215)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient401)
  (cocktail-part2 cocktail352 ingredient129)
)
 (:goal
  (and
      (contains shot131 cocktail352)
      (contains shot215 cocktail352)
)))
