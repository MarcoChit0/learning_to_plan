(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot101 shot380 - shot
      ingredient266 ingredient409 - ingredient
      cocktail79 - cocktail
      dispenser53 dispenser235 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot101)
  (ontable shot380)
  (dispenses dispenser53 ingredient266)
  (dispenses dispenser235 ingredient409)
  (clean shaker332)
  (clean shot101)
  (clean shot380)
  (empty shaker332)
  (empty shot101)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail79 ingredient266)
  (cocktail-part2 cocktail79 ingredient409)
)
 (:goal
  (and
      (contains shot101 cocktail79)
)))
