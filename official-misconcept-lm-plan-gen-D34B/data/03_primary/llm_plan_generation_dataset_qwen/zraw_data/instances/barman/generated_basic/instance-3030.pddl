(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot298 - shot
      ingredient393 ingredient37 ingredient175 - ingredient
      cocktail201 - cocktail
      dispenser343 dispenser497 dispenser499 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot298)
  (dispenses dispenser343 ingredient393)
  (dispenses dispenser497 ingredient37)
  (dispenses dispenser499 ingredient175)
  (clean shaker146)
  (clean shot298)
  (empty shaker146)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail201 ingredient175)
  (cocktail-part2 cocktail201 ingredient393)
)
 (:goal
  (and
      (contains shot298 cocktail201)
)))
