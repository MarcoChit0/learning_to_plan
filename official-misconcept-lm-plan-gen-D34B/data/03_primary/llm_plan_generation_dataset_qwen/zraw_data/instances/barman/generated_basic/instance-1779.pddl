(define (problem prob)
 (:domain barman)
 (:objects 
      shaker223 - shaker
      left right - hand
      shot304 shot331 - shot
      ingredient51 ingredient104 ingredient219 - ingredient
      cocktail461 - cocktail
      dispenser495 dispenser109 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker223)
  (ontable shot304)
  (ontable shot331)
  (dispenses dispenser495 ingredient51)
  (dispenses dispenser109 ingredient104)
  (dispenses dispenser132 ingredient219)
  (clean shaker223)
  (clean shot304)
  (clean shot331)
  (empty shaker223)
  (empty shot304)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker223 l0)
  (shaker-level shaker223 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient104)
  (cocktail-part2 cocktail461 ingredient219)
)
 (:goal
  (and
      (contains shot304 cocktail461)
)))
