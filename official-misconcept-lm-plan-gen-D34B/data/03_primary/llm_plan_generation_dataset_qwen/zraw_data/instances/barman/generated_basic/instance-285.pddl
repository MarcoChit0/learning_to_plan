(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot321 shot234 - shot
      ingredient274 ingredient304 ingredient214 - ingredient
      cocktail1 - cocktail
      dispenser438 dispenser167 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot321)
  (ontable shot234)
  (dispenses dispenser438 ingredient274)
  (dispenses dispenser167 ingredient304)
  (dispenses dispenser481 ingredient214)
  (clean shaker439)
  (clean shot321)
  (clean shot234)
  (empty shaker439)
  (empty shot321)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient304)
  (cocktail-part2 cocktail1 ingredient274)
)
 (:goal
  (and
      (contains shot321 cocktail1)
)))
