(define (problem prob)
 (:domain barman)
 (:objects 
      shaker245 - shaker
      left right - hand
      shot496 - shot
      ingredient272 ingredient166 - ingredient
      cocktail75 - cocktail
      dispenser416 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker245)
  (ontable shot496)
  (dispenses dispenser416 ingredient272)
  (dispenses dispenser285 ingredient166)
  (clean shaker245)
  (clean shot496)
  (empty shaker245)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker245 l0)
  (shaker-level shaker245 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient272)
  (cocktail-part2 cocktail75 ingredient166)
)
 (:goal
  (and
      (contains shot496 cocktail75)
)))
