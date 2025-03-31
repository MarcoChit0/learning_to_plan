(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot489 - shot
      ingredient103 ingredient200 ingredient246 - ingredient
      cocktail1 - cocktail
      dispenser16 dispenser401 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot489)
  (dispenses dispenser16 ingredient103)
  (dispenses dispenser401 ingredient200)
  (dispenses dispenser343 ingredient246)
  (clean shaker281)
  (clean shot489)
  (empty shaker281)
  (empty shot489)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient200)
  (cocktail-part2 cocktail1 ingredient246)
)
 (:goal
  (and
      (contains shot489 cocktail1)
)))
