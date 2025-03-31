(define (problem prob)
 (:domain barman)
 (:objects 
      shaker217 - shaker
      left right - hand
      shot95 - shot
      ingredient369 ingredient274 - ingredient
      cocktail305 - cocktail
      dispenser310 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker217)
  (ontable shot95)
  (dispenses dispenser310 ingredient369)
  (dispenses dispenser122 ingredient274)
  (clean shaker217)
  (clean shot95)
  (empty shaker217)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker217 l0)
  (shaker-level shaker217 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient274)
  (cocktail-part2 cocktail305 ingredient369)
)
 (:goal
  (and
      (contains shot95 cocktail305)
)))
