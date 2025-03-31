(define (problem prob)
 (:domain barman)
 (:objects 
      shaker493 - shaker
      left right - hand
      shot287 - shot
      ingredient169 ingredient159 - ingredient
      cocktail1 - cocktail
      dispenser468 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker493)
  (ontable shot287)
  (dispenses dispenser468 ingredient169)
  (dispenses dispenser248 ingredient159)
  (clean shaker493)
  (clean shot287)
  (empty shaker493)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker493 l0)
  (shaker-level shaker493 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient169)
  (cocktail-part2 cocktail1 ingredient159)
)
 (:goal
  (and
      (contains shot287 cocktail1)
)))
