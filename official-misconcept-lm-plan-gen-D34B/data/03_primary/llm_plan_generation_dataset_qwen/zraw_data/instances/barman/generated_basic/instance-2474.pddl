(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot280 - shot
      ingredient177 ingredient378 ingredient250 - ingredient
      cocktail311 - cocktail
      dispenser315 dispenser38 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot280)
  (dispenses dispenser315 ingredient177)
  (dispenses dispenser38 ingredient378)
  (dispenses dispenser387 ingredient250)
  (clean shaker186)
  (clean shot280)
  (empty shaker186)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient378)
  (cocktail-part2 cocktail311 ingredient177)
)
 (:goal
  (and
      (contains shot280 cocktail311)
)))
