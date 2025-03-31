(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot141 shot93 - shot
      ingredient350 ingredient159 - ingredient
      cocktail1 - cocktail
      dispenser109 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot141)
  (ontable shot93)
  (dispenses dispenser109 ingredient350)
  (dispenses dispenser472 ingredient159)
  (clean shaker152)
  (clean shot141)
  (clean shot93)
  (empty shaker152)
  (empty shot141)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient159)
  (cocktail-part2 cocktail1 ingredient350)
)
 (:goal
  (and
      (contains shot141 cocktail1)
)))
