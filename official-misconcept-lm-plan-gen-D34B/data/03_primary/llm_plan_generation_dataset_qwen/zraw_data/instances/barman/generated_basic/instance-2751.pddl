(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot233 - shot
      ingredient8 ingredient7 - ingredient
      cocktail357 - cocktail
      dispenser167 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot233)
  (dispenses dispenser167 ingredient8)
  (dispenses dispenser315 ingredient7)
  (clean shaker29)
  (clean shot233)
  (empty shaker29)
  (empty shot233)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient7)
  (cocktail-part2 cocktail357 ingredient8)
)
 (:goal
  (and
      (contains shot233 cocktail357)
)))
