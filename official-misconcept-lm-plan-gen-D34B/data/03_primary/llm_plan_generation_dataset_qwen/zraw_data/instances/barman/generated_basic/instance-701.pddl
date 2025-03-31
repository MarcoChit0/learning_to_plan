(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot62 shot432 - shot
      ingredient378 ingredient356 ingredient429 - ingredient
      cocktail1 - cocktail
      dispenser212 dispenser332 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot62)
  (ontable shot432)
  (dispenses dispenser212 ingredient378)
  (dispenses dispenser332 ingredient356)
  (dispenses dispenser360 ingredient429)
  (clean shaker112)
  (clean shot62)
  (clean shot432)
  (empty shaker112)
  (empty shot62)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient429)
  (cocktail-part2 cocktail1 ingredient356)
)
 (:goal
  (and
      (contains shot62 cocktail1)
)))
