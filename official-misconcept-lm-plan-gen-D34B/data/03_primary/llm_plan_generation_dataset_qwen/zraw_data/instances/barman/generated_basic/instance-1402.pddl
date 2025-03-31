(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot421 - shot
      ingredient134 ingredient121 ingredient240 ingredient89 - ingredient
      cocktail83 - cocktail
      dispenser347 dispenser237 dispenser358 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot421)
  (dispenses dispenser347 ingredient134)
  (dispenses dispenser237 ingredient121)
  (dispenses dispenser358 ingredient240)
  (dispenses dispenser336 ingredient89)
  (clean shaker15)
  (clean shot421)
  (empty shaker15)
  (empty shot421)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient121)
  (cocktail-part2 cocktail83 ingredient240)
)
 (:goal
  (and
      (contains shot421 cocktail83)
)))
