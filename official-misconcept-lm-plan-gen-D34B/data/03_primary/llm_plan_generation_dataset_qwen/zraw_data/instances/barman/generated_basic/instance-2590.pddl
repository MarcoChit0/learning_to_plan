(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot127 - shot
      ingredient368 ingredient32 - ingredient
      cocktail471 - cocktail
      dispenser151 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot127)
  (dispenses dispenser151 ingredient368)
  (dispenses dispenser216 ingredient32)
  (clean shaker372)
  (clean shot127)
  (empty shaker372)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail471 ingredient368)
  (cocktail-part2 cocktail471 ingredient32)
)
 (:goal
  (and
      (contains shot127 cocktail471)
)))
