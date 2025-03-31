(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot411 - shot
      ingredient200 ingredient267 ingredient356 ingredient20 - ingredient
      cocktail454 - cocktail
      dispenser23 dispenser52 dispenser322 dispenser1 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot411)
  (dispenses dispenser23 ingredient200)
  (dispenses dispenser52 ingredient267)
  (dispenses dispenser322 ingredient356)
  (dispenses dispenser1 ingredient20)
  (clean shaker113)
  (clean shot411)
  (empty shaker113)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail454 ingredient356)
  (cocktail-part2 cocktail454 ingredient267)
)
 (:goal
  (and
      (contains shot411 cocktail454)
)))
