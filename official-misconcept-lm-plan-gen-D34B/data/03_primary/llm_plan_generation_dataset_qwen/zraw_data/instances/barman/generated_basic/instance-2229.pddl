(define (problem prob)
 (:domain barman)
 (:objects 
      shaker253 - shaker
      left right - hand
      shot56 - shot
      ingredient43 ingredient18 - ingredient
      cocktail48 - cocktail
      dispenser247 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker253)
  (ontable shot56)
  (dispenses dispenser247 ingredient43)
  (dispenses dispenser71 ingredient18)
  (clean shaker253)
  (clean shot56)
  (empty shaker253)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker253 l0)
  (shaker-level shaker253 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient18)
  (cocktail-part2 cocktail48 ingredient43)
)
 (:goal
  (and
      (contains shot56 cocktail48)
)))
