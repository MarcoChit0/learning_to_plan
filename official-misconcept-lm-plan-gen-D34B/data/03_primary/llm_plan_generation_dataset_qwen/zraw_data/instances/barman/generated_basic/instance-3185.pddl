(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot25 - shot
      ingredient400 ingredient182 ingredient150 - ingredient
      cocktail354 - cocktail
      dispenser186 dispenser393 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot25)
  (dispenses dispenser186 ingredient400)
  (dispenses dispenser393 ingredient182)
  (dispenses dispenser290 ingredient150)
  (clean shaker312)
  (clean shot25)
  (empty shaker312)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail354 ingredient150)
  (cocktail-part2 cocktail354 ingredient400)
)
 (:goal
  (and
      (contains shot25 cocktail354)
)))
