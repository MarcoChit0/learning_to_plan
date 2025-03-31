(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot294 - shot
      ingredient57 ingredient446 - ingredient
      cocktail1 - cocktail
      dispenser34 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot294)
  (dispenses dispenser34 ingredient57)
  (dispenses dispenser321 ingredient446)
  (clean shaker298)
  (clean shot294)
  (empty shaker298)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient446)
  (cocktail-part2 cocktail1 ingredient57)
)
 (:goal
  (and
      (contains shot294 cocktail1)
)))
