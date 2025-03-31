(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot67 shot332 - shot
      ingredient408 ingredient461 ingredient434 - ingredient
      cocktail291 - cocktail
      dispenser89 dispenser197 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot67)
  (ontable shot332)
  (dispenses dispenser89 ingredient408)
  (dispenses dispenser197 ingredient461)
  (dispenses dispenser293 ingredient434)
  (clean shaker114)
  (clean shot67)
  (clean shot332)
  (empty shaker114)
  (empty shot67)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient461)
  (cocktail-part2 cocktail291 ingredient408)
)
 (:goal
  (and
      (contains shot67 cocktail291)
)))
