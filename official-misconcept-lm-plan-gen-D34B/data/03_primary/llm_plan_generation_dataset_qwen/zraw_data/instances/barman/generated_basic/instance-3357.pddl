(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot233 - shot
      ingredient121 ingredient274 - ingredient
      cocktail83 - cocktail
      dispenser341 dispenser94 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot233)
  (dispenses dispenser341 ingredient121)
  (dispenses dispenser94 ingredient274)
  (clean shaker272)
  (clean shot233)
  (empty shaker272)
  (empty shot233)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient274)
  (cocktail-part2 cocktail83 ingredient121)
)
 (:goal
  (and
      (contains shot233 cocktail83)
)))
