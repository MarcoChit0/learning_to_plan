(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot46 - shot
      ingredient260 ingredient110 - ingredient
      cocktail1 - cocktail
      dispenser354 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot46)
  (dispenses dispenser354 ingredient260)
  (dispenses dispenser40 ingredient110)
  (clean shaker156)
  (clean shot46)
  (empty shaker156)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient260)
  (cocktail-part2 cocktail1 ingredient110)
)
 (:goal
  (and
      (contains shot46 cocktail1)
)))
