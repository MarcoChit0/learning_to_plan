(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot395 - shot
      ingredient347 ingredient190 - ingredient
      cocktail371 - cocktail
      dispenser127 dispenser401 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot395)
  (dispenses dispenser127 ingredient347)
  (dispenses dispenser401 ingredient190)
  (clean shaker221)
  (clean shot395)
  (empty shaker221)
  (empty shot395)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient347)
  (cocktail-part2 cocktail371 ingredient190)
)
 (:goal
  (and
      (contains shot395 cocktail371)
)))
