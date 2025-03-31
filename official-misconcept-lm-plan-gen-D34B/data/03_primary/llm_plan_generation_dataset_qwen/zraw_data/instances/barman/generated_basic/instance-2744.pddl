(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot465 - shot
      ingredient81 ingredient154 - ingredient
      cocktail36 - cocktail
      dispenser184 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot465)
  (dispenses dispenser184 ingredient81)
  (dispenses dispenser155 ingredient154)
  (clean shaker403)
  (clean shot465)
  (empty shaker403)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient154)
  (cocktail-part2 cocktail36 ingredient81)
)
 (:goal
  (and
      (contains shot465 cocktail36)
)))
