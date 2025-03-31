(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot439 - shot
      ingredient274 ingredient106 ingredient8 ingredient352 - ingredient
      cocktail68 - cocktail
      dispenser440 dispenser184 dispenser105 dispenser379 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot439)
  (dispenses dispenser440 ingredient274)
  (dispenses dispenser184 ingredient106)
  (dispenses dispenser105 ingredient8)
  (dispenses dispenser379 ingredient352)
  (clean shaker250)
  (clean shot439)
  (empty shaker250)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail68 ingredient274)
  (cocktail-part2 cocktail68 ingredient106)
)
 (:goal
  (and
      (contains shot439 cocktail68)
)))
