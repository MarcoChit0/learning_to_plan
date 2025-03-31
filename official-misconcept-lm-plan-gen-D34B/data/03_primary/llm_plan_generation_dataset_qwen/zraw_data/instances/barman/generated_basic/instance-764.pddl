(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot401 - shot
      ingredient197 ingredient438 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot401)
  (dispenses dispenser248 ingredient197)
  (dispenses dispenser369 ingredient438)
  (clean shaker437)
  (clean shot401)
  (empty shaker437)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient438)
  (cocktail-part2 cocktail1 ingredient197)
)
 (:goal
  (and
      (contains shot401 cocktail1)
)))
