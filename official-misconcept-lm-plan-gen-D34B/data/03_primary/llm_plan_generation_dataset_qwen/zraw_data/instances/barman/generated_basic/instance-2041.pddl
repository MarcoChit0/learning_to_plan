(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot25 - shot
      ingredient228 ingredient56 ingredient406 ingredient132 - ingredient
      cocktail498 - cocktail
      dispenser343 dispenser364 dispenser352 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot25)
  (dispenses dispenser343 ingredient228)
  (dispenses dispenser364 ingredient56)
  (dispenses dispenser352 ingredient406)
  (dispenses dispenser128 ingredient132)
  (clean shaker238)
  (clean shot25)
  (empty shaker238)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail498 ingredient56)
  (cocktail-part2 cocktail498 ingredient406)
)
 (:goal
  (and
      (contains shot25 cocktail498)
)))
