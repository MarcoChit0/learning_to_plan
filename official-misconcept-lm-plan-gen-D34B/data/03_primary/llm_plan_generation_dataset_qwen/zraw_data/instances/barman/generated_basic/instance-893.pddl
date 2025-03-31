(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot39 shot218 - shot
      ingredient491 ingredient50 - ingredient
      cocktail1 - cocktail
      dispenser199 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot39)
  (ontable shot218)
  (dispenses dispenser199 ingredient491)
  (dispenses dispenser311 ingredient50)
  (clean shaker6)
  (clean shot39)
  (clean shot218)
  (empty shaker6)
  (empty shot39)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient491)
  (cocktail-part2 cocktail1 ingredient50)
)
 (:goal
  (and
      (contains shot39 cocktail1)
)))
