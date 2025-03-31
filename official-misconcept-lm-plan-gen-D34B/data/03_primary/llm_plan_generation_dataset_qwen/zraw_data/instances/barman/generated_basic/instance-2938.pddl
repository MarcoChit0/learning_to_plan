(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot282 - shot
      ingredient242 ingredient205 - ingredient
      cocktail2 - cocktail
      dispenser328 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot282)
  (dispenses dispenser328 ingredient242)
  (dispenses dispenser242 ingredient205)
  (clean shaker174)
  (clean shot282)
  (empty shaker174)
  (empty shot282)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient242)
  (cocktail-part2 cocktail2 ingredient205)
)
 (:goal
  (and
      (contains shot282 cocktail2)
)))
