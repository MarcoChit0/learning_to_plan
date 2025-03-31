(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot394 - shot
      ingredient35 ingredient469 - ingredient
      cocktail346 - cocktail
      dispenser62 dispenser450 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot394)
  (dispenses dispenser62 ingredient35)
  (dispenses dispenser450 ingredient469)
  (clean shaker409)
  (clean shot394)
  (empty shaker409)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient35)
  (cocktail-part2 cocktail346 ingredient469)
)
 (:goal
  (and
      (contains shot394 cocktail346)
)))
