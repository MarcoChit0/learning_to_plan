(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot197 - shot
      ingredient95 ingredient25 ingredient220 - ingredient
      cocktail1 - cocktail
      dispenser393 dispenser143 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot197)
  (dispenses dispenser393 ingredient95)
  (dispenses dispenser143 ingredient25)
  (dispenses dispenser11 ingredient220)
  (clean shaker24)
  (clean shot197)
  (empty shaker24)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient25)
  (cocktail-part2 cocktail1 ingredient220)
)
 (:goal
  (and
      (contains shot197 cocktail1)
)))
