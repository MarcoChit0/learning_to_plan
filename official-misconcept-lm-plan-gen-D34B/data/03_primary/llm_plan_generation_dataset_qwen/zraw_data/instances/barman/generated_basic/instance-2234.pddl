(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot499 - shot
      ingredient408 ingredient150 - ingredient
      cocktail223 - cocktail
      dispenser39 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot499)
  (dispenses dispenser39 ingredient408)
  (dispenses dispenser431 ingredient150)
  (clean shaker453)
  (clean shot499)
  (empty shaker453)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient150)
  (cocktail-part2 cocktail223 ingredient408)
)
 (:goal
  (and
      (contains shot499 cocktail223)
)))
