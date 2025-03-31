(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot245 - shot
      ingredient460 ingredient240 - ingredient
      cocktail50 - cocktail
      dispenser271 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot245)
  (dispenses dispenser271 ingredient460)
  (dispenses dispenser409 ingredient240)
  (clean shaker113)
  (clean shot245)
  (empty shaker113)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail50 ingredient460)
  (cocktail-part2 cocktail50 ingredient240)
)
 (:goal
  (and
      (contains shot245 cocktail50)
)))
