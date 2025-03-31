(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot55 - shot
      ingredient497 ingredient115 - ingredient
      cocktail449 - cocktail
      dispenser178 dispenser401 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot55)
  (dispenses dispenser178 ingredient497)
  (dispenses dispenser401 ingredient115)
  (clean shaker5)
  (clean shot55)
  (empty shaker5)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient497)
  (cocktail-part2 cocktail449 ingredient115)
)
 (:goal
  (and
      (contains shot55 cocktail449)
)))
