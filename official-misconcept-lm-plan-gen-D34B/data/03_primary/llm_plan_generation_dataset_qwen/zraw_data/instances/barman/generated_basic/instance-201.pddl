(define (problem prob)
 (:domain barman)
 (:objects 
      shaker423 - shaker
      left right - hand
      shot364 - shot
      ingredient194 ingredient123 ingredient45 - ingredient
      cocktail1 - cocktail
      dispenser360 dispenser129 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker423)
  (ontable shot364)
  (dispenses dispenser360 ingredient194)
  (dispenses dispenser129 ingredient123)
  (dispenses dispenser472 ingredient45)
  (clean shaker423)
  (clean shot364)
  (empty shaker423)
  (empty shot364)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker423 l0)
  (shaker-level shaker423 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient194)
  (cocktail-part2 cocktail1 ingredient123)
)
 (:goal
  (and
      (contains shot364 cocktail1)
)))
