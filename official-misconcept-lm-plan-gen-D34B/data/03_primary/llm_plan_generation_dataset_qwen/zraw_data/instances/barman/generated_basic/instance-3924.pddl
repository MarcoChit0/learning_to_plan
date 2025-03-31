(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot73 - shot
      ingredient284 ingredient404 ingredient7 ingredient112 - ingredient
      cocktail207 - cocktail
      dispenser393 dispenser368 dispenser380 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot73)
  (dispenses dispenser393 ingredient284)
  (dispenses dispenser368 ingredient404)
  (dispenses dispenser380 ingredient7)
  (dispenses dispenser22 ingredient112)
  (clean shaker211)
  (clean shot73)
  (empty shaker211)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient404)
  (cocktail-part2 cocktail207 ingredient112)
)
 (:goal
  (and
      (contains shot73 cocktail207)
)))
