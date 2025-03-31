(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot83 shot46 shot14 - shot
      ingredient117 ingredient274 - ingredient
      cocktail104 - cocktail
      dispenser120 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot83)
  (ontable shot46)
  (ontable shot14)
  (dispenses dispenser120 ingredient117)
  (dispenses dispenser244 ingredient274)
  (clean shaker167)
  (clean shot83)
  (clean shot46)
  (clean shot14)
  (empty shaker167)
  (empty shot83)
  (empty shot46)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient117)
  (cocktail-part2 cocktail104 ingredient274)
)
 (:goal
  (and
      (contains shot83 cocktail104)
      (contains shot46 cocktail104)
)))
