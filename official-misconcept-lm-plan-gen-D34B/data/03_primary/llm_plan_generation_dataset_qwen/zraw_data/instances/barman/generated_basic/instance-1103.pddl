(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot496 shot168 - shot
      ingredient38 ingredient378 - ingredient
      cocktail1 - cocktail
      dispenser437 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot496)
  (ontable shot168)
  (dispenses dispenser437 ingredient38)
  (dispenses dispenser150 ingredient378)
  (clean shaker409)
  (clean shot496)
  (clean shot168)
  (empty shaker409)
  (empty shot496)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient38)
  (cocktail-part2 cocktail1 ingredient378)
)
 (:goal
  (and
      (contains shot496 cocktail1)
)))
