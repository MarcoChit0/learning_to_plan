(define (problem prob)
 (:domain barman)
 (:objects 
      shaker473 - shaker
      left right - hand
      shot210 - shot
      ingredient435 ingredient368 ingredient336 - ingredient
      cocktail28 - cocktail
      dispenser331 dispenser388 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker473)
  (ontable shot210)
  (dispenses dispenser331 ingredient435)
  (dispenses dispenser388 ingredient368)
  (dispenses dispenser363 ingredient336)
  (clean shaker473)
  (clean shot210)
  (empty shaker473)
  (empty shot210)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker473 l0)
  (shaker-level shaker473 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail28 ingredient336)
  (cocktail-part2 cocktail28 ingredient368)
)
 (:goal
  (and
      (contains shot210 cocktail28)
)))
