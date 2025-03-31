(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot371 - shot
      ingredient403 ingredient81 - ingredient
      cocktail447 - cocktail
      dispenser34 dispenser105 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot371)
  (dispenses dispenser34 ingredient403)
  (dispenses dispenser105 ingredient81)
  (clean shaker326)
  (clean shot371)
  (empty shaker326)
  (empty shot371)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail447 ingredient81)
  (cocktail-part2 cocktail447 ingredient403)
)
 (:goal
  (and
      (contains shot371 cocktail447)
)))
