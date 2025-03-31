(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot244 - shot
      ingredient374 ingredient427 ingredient278 ingredient432 - ingredient
      cocktail53 - cocktail
      dispenser458 dispenser126 dispenser481 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot244)
  (dispenses dispenser458 ingredient374)
  (dispenses dispenser126 ingredient427)
  (dispenses dispenser481 ingredient278)
  (dispenses dispenser352 ingredient432)
  (clean shaker484)
  (clean shot244)
  (empty shaker484)
  (empty shot244)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient432)
  (cocktail-part2 cocktail53 ingredient374)
)
 (:goal
  (and
      (contains shot244 cocktail53)
)))
