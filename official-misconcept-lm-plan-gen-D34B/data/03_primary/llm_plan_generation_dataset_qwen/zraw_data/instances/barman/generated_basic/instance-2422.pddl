(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot174 - shot
      ingredient450 ingredient370 - ingredient
      cocktail328 - cocktail
      dispenser387 dispenser181 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot174)
  (dispenses dispenser387 ingredient450)
  (dispenses dispenser181 ingredient370)
  (clean shaker184)
  (clean shot174)
  (empty shaker184)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient370)
  (cocktail-part2 cocktail328 ingredient450)
)
 (:goal
  (and
      (contains shot174 cocktail328)
)))
