(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot46 - shot
      ingredient257 ingredient21 ingredient214 - ingredient
      cocktail496 - cocktail
      dispenser359 dispenser459 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot46)
  (dispenses dispenser359 ingredient257)
  (dispenses dispenser459 ingredient21)
  (dispenses dispenser273 ingredient214)
  (clean shaker373)
  (clean shot46)
  (empty shaker373)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail496 ingredient214)
  (cocktail-part2 cocktail496 ingredient21)
)
 (:goal
  (and
      (contains shot46 cocktail496)
)))
