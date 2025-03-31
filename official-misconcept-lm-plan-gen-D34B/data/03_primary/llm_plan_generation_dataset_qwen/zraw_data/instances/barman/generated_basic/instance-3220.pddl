(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot343 - shot
      ingredient442 ingredient226 - ingredient
      cocktail175 - cocktail
      dispenser422 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot343)
  (dispenses dispenser422 ingredient442)
  (dispenses dispenser44 ingredient226)
  (clean shaker373)
  (clean shot343)
  (empty shaker373)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail175 ingredient226)
  (cocktail-part2 cocktail175 ingredient442)
)
 (:goal
  (and
      (contains shot343 cocktail175)
)))
