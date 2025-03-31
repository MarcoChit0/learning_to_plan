(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot332 - shot
      ingredient442 ingredient319 - ingredient
      cocktail310 - cocktail
      dispenser396 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot332)
  (dispenses dispenser396 ingredient442)
  (dispenses dispenser275 ingredient319)
  (clean shaker219)
  (clean shot332)
  (empty shaker219)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail310 ingredient319)
  (cocktail-part2 cocktail310 ingredient442)
)
 (:goal
  (and
      (contains shot332 cocktail310)
)))
