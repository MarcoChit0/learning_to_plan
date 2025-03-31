(define (problem prob)
 (:domain barman)
 (:objects 
      shaker118 - shaker
      left right - hand
      shot305 - shot
      ingredient466 ingredient430 - ingredient
      cocktail1 - cocktail
      dispenser437 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker118)
  (ontable shot305)
  (dispenses dispenser437 ingredient466)
  (dispenses dispenser61 ingredient430)
  (clean shaker118)
  (clean shot305)
  (empty shaker118)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker118 l0)
  (shaker-level shaker118 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient430)
  (cocktail-part2 cocktail1 ingredient466)
)
 (:goal
  (and
      (contains shot305 cocktail1)
)))
