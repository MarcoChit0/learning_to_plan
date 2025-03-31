(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot152 - shot
      ingredient194 ingredient277 - ingredient
      cocktail1 - cocktail
      dispenser80 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot152)
  (dispenses dispenser80 ingredient194)
  (dispenses dispenser193 ingredient277)
  (clean shaker55)
  (clean shot152)
  (empty shaker55)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient277)
  (cocktail-part2 cocktail1 ingredient194)
)
 (:goal
  (and
      (contains shot152 cocktail1)
)))
