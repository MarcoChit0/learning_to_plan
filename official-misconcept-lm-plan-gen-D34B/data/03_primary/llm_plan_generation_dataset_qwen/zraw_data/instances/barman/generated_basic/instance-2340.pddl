(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot411 - shot
      ingredient497 ingredient324 - ingredient
      cocktail11 - cocktail
      dispenser287 dispenser452 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot411)
  (dispenses dispenser287 ingredient497)
  (dispenses dispenser452 ingredient324)
  (clean shaker42)
  (clean shot411)
  (empty shaker42)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail11 ingredient497)
  (cocktail-part2 cocktail11 ingredient324)
)
 (:goal
  (and
      (contains shot411 cocktail11)
)))
