(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot77 - shot
      ingredient197 ingredient462 ingredient237 - ingredient
      cocktail1 - cocktail
      dispenser294 dispenser328 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot77)
  (dispenses dispenser294 ingredient197)
  (dispenses dispenser328 ingredient462)
  (dispenses dispenser305 ingredient237)
  (clean shaker174)
  (clean shot77)
  (empty shaker174)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient197)
  (cocktail-part2 cocktail1 ingredient462)
)
 (:goal
  (and
      (contains shot77 cocktail1)
)))
