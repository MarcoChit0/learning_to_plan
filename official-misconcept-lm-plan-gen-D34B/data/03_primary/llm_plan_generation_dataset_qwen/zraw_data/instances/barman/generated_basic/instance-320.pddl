(define (problem prob)
 (:domain barman)
 (:objects 
      shaker23 - shaker
      left right - hand
      shot295 - shot
      ingredient383 ingredient163 - ingredient
      cocktail1 - cocktail
      dispenser213 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker23)
  (ontable shot295)
  (dispenses dispenser213 ingredient383)
  (dispenses dispenser114 ingredient163)
  (clean shaker23)
  (clean shot295)
  (empty shaker23)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker23 l0)
  (shaker-level shaker23 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient163)
  (cocktail-part2 cocktail1 ingredient383)
)
 (:goal
  (and
      (contains shot295 cocktail1)
)))
