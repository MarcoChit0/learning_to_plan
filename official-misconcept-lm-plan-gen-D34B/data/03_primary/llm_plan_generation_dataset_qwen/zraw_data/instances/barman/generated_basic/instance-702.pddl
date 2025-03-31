(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot39 shot141 - shot
      ingredient233 ingredient190 ingredient484 - ingredient
      cocktail1 - cocktail
      dispenser0 dispenser222 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot39)
  (ontable shot141)
  (dispenses dispenser0 ingredient233)
  (dispenses dispenser222 ingredient190)
  (dispenses dispenser293 ingredient484)
  (clean shaker209)
  (clean shot39)
  (clean shot141)
  (empty shaker209)
  (empty shot39)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient233)
  (cocktail-part2 cocktail1 ingredient484)
)
 (:goal
  (and
      (contains shot39 cocktail1)
)))
