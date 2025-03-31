(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot155 - shot
      ingredient450 ingredient306 - ingredient
      cocktail1 - cocktail
      dispenser285 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot155)
  (dispenses dispenser285 ingredient450)
  (dispenses dispenser406 ingredient306)
  (clean shaker244)
  (clean shot155)
  (empty shaker244)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient306)
  (cocktail-part2 cocktail1 ingredient450)
)
 (:goal
  (and
      (contains shot155 cocktail1)
)))
