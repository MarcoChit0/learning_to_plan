(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot150 shot179 shot341 - shot
      ingredient274 ingredient454 ingredient407 - ingredient
      cocktail30 - cocktail
      dispenser476 dispenser120 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot150)
  (ontable shot179)
  (ontable shot341)
  (dispenses dispenser476 ingredient274)
  (dispenses dispenser120 ingredient454)
  (dispenses dispenser29 ingredient407)
  (clean shaker31)
  (clean shot150)
  (clean shot179)
  (clean shot341)
  (empty shaker31)
  (empty shot150)
  (empty shot179)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient274)
  (cocktail-part2 cocktail30 ingredient454)
)
 (:goal
  (and
      (contains shot150 cocktail30)
      (contains shot179 cocktail30)
)))
