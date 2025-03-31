(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot190 shot479 - shot
      ingredient55 ingredient426 ingredient324 - ingredient
      cocktail147 - cocktail
      dispenser108 dispenser73 dispenser12 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot190)
  (ontable shot479)
  (dispenses dispenser108 ingredient55)
  (dispenses dispenser73 ingredient426)
  (dispenses dispenser12 ingredient324)
  (clean shaker380)
  (clean shot190)
  (clean shot479)
  (empty shaker380)
  (empty shot190)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail147 ingredient426)
  (cocktail-part2 cocktail147 ingredient324)
)
 (:goal
  (and
      (contains shot190 cocktail147)
)))
