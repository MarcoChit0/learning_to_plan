(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot466 shot422 - shot
      ingredient408 ingredient259 ingredient9 - ingredient
      cocktail392 - cocktail
      dispenser209 dispenser405 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot466)
  (ontable shot422)
  (dispenses dispenser209 ingredient408)
  (dispenses dispenser405 ingredient259)
  (dispenses dispenser143 ingredient9)
  (clean shaker7)
  (clean shot466)
  (clean shot422)
  (empty shaker7)
  (empty shot466)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail392 ingredient9)
  (cocktail-part2 cocktail392 ingredient259)
)
 (:goal
  (and
      (contains shot466 cocktail392)
)))
