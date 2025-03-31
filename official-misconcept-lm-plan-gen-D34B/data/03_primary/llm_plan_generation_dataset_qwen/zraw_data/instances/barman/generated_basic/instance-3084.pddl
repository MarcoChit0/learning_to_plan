(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot75 shot25 shot138 shot304 - shot
      ingredient425 ingredient66 ingredient192 ingredient466 - ingredient
      cocktail360 - cocktail
      dispenser108 dispenser39 dispenser114 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot75)
  (ontable shot25)
  (ontable shot138)
  (ontable shot304)
  (dispenses dispenser108 ingredient425)
  (dispenses dispenser39 ingredient66)
  (dispenses dispenser114 ingredient192)
  (dispenses dispenser487 ingredient466)
  (clean shaker237)
  (clean shot75)
  (clean shot25)
  (clean shot138)
  (clean shot304)
  (empty shaker237)
  (empty shot75)
  (empty shot25)
  (empty shot138)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient466)
  (cocktail-part2 cocktail360 ingredient425)
)
 (:goal
  (and
      (contains shot75 cocktail360)
      (contains shot25 cocktail360)
      (contains shot138 ingredient466)
)))
