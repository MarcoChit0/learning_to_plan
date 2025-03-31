(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot292 - shot
      ingredient465 ingredient372 - ingredient
      cocktail1 - cocktail
      dispenser204 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot292)
  (dispenses dispenser204 ingredient465)
  (dispenses dispenser183 ingredient372)
  (clean shaker200)
  (clean shot292)
  (empty shaker200)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient465)
  (cocktail-part2 cocktail1 ingredient372)
)
 (:goal
  (and
      (contains shot292 cocktail1)
)))
