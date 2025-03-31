(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot197 - shot
      ingredient244 ingredient72 ingredient460 ingredient387 - ingredient
      cocktail375 - cocktail
      dispenser222 dispenser453 dispenser153 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot197)
  (dispenses dispenser222 ingredient244)
  (dispenses dispenser453 ingredient72)
  (dispenses dispenser153 ingredient460)
  (dispenses dispenser33 ingredient387)
  (clean shaker168)
  (clean shot197)
  (empty shaker168)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient72)
  (cocktail-part2 cocktail375 ingredient387)
)
 (:goal
  (and
      (contains shot197 cocktail375)
)))
