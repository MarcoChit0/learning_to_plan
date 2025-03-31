(define (problem prob)
 (:domain barman)
 (:objects 
      shaker134 - shaker
      left right - hand
      shot375 - shot
      ingredient116 ingredient242 - ingredient
      cocktail385 - cocktail
      dispenser338 dispenser379 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker134)
  (ontable shot375)
  (dispenses dispenser338 ingredient116)
  (dispenses dispenser379 ingredient242)
  (clean shaker134)
  (clean shot375)
  (empty shaker134)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker134 l0)
  (shaker-level shaker134 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail385 ingredient242)
  (cocktail-part2 cocktail385 ingredient116)
)
 (:goal
  (and
      (contains shot375 cocktail385)
)))
