(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot497 - shot
      ingredient375 ingredient57 - ingredient
      cocktail328 - cocktail
      dispenser314 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot497)
  (dispenses dispenser314 ingredient375)
  (dispenses dispenser466 ingredient57)
  (clean shaker18)
  (clean shot497)
  (empty shaker18)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient375)
  (cocktail-part2 cocktail328 ingredient57)
)
 (:goal
  (and
      (contains shot497 cocktail328)
)))
