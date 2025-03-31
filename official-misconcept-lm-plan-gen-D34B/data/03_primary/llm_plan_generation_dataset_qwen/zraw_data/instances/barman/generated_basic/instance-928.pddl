(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot46 shot354 - shot
      ingredient14 ingredient381 ingredient104 - ingredient
      cocktail1 - cocktail
      dispenser182 dispenser378 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot46)
  (ontable shot354)
  (dispenses dispenser182 ingredient14)
  (dispenses dispenser378 ingredient381)
  (dispenses dispenser474 ingredient104)
  (clean shaker380)
  (clean shot46)
  (clean shot354)
  (empty shaker380)
  (empty shot46)
  (empty shot354)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient381)
  (cocktail-part2 cocktail1 ingredient104)
)
 (:goal
  (and
      (contains shot46 cocktail1)
)))
