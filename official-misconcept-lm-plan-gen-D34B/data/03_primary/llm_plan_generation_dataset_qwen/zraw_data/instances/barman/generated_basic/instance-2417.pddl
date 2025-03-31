(define (problem prob)
 (:domain barman)
 (:objects 
      shaker208 - shaker
      left right - hand
      shot152 - shot
      ingredient472 ingredient395 - ingredient
      cocktail201 - cocktail
      dispenser216 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker208)
  (ontable shot152)
  (dispenses dispenser216 ingredient472)
  (dispenses dispenser179 ingredient395)
  (clean shaker208)
  (clean shot152)
  (empty shaker208)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker208 l0)
  (shaker-level shaker208 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail201 ingredient472)
  (cocktail-part2 cocktail201 ingredient395)
)
 (:goal
  (and
      (contains shot152 cocktail201)
)))
