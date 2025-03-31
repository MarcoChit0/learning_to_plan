(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot459 - shot
      ingredient123 ingredient444 - ingredient
      cocktail196 - cocktail
      dispenser206 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot459)
  (dispenses dispenser206 ingredient123)
  (dispenses dispenser399 ingredient444)
  (clean shaker187)
  (clean shot459)
  (empty shaker187)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail196 ingredient123)
  (cocktail-part2 cocktail196 ingredient444)
)
 (:goal
  (and
      (contains shot459 cocktail196)
)))
