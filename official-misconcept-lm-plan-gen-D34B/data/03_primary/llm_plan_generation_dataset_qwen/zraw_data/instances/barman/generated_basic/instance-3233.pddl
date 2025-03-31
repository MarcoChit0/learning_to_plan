(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot136 - shot
      ingredient35 ingredient340 ingredient495 - ingredient
      cocktail291 - cocktail
      dispenser165 dispenser460 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot136)
  (dispenses dispenser165 ingredient35)
  (dispenses dispenser460 ingredient340)
  (dispenses dispenser355 ingredient495)
  (clean shaker332)
  (clean shot136)
  (empty shaker332)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient35)
  (cocktail-part2 cocktail291 ingredient340)
)
 (:goal
  (and
      (contains shot136 cocktail291)
)))
