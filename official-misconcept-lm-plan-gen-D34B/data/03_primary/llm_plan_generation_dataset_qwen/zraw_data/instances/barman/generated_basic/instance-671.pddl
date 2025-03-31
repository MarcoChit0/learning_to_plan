(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot128 shot42 - shot
      ingredient194 ingredient173 - ingredient
      cocktail1 - cocktail
      dispenser168 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot128)
  (ontable shot42)
  (dispenses dispenser168 ingredient194)
  (dispenses dispenser196 ingredient173)
  (clean shaker429)
  (clean shot128)
  (clean shot42)
  (empty shaker429)
  (empty shot128)
  (empty shot42)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient173)
  (cocktail-part2 cocktail1 ingredient194)
)
 (:goal
  (and
      (contains shot128 cocktail1)
)))
