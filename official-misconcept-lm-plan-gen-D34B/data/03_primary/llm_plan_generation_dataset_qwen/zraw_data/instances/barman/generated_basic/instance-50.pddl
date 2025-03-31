(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot499 shot292 - shot
      ingredient366 ingredient128 ingredient422 ingredient131 - ingredient
      cocktail1 - cocktail
      dispenser211 dispenser33 dispenser297 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot499)
  (ontable shot292)
  (dispenses dispenser211 ingredient366)
  (dispenses dispenser33 ingredient128)
  (dispenses dispenser297 ingredient422)
  (dispenses dispenser290 ingredient131)
  (clean shaker104)
  (clean shot499)
  (clean shot292)
  (empty shaker104)
  (empty shot499)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient128)
  (cocktail-part2 cocktail1 ingredient131)
)
 (:goal
  (and
      (contains shot499 cocktail1)
)))
