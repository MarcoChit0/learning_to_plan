(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot167 - shot
      ingredient218 ingredient176 ingredient248 - ingredient
      cocktail1 - cocktail
      dispenser93 dispenser343 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot167)
  (dispenses dispenser93 ingredient218)
  (dispenses dispenser343 ingredient176)
  (dispenses dispenser72 ingredient248)
  (clean shaker395)
  (clean shot167)
  (empty shaker395)
  (empty shot167)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient248)
  (cocktail-part2 cocktail1 ingredient176)
)
 (:goal
  (and
      (contains shot167 cocktail1)
)))
