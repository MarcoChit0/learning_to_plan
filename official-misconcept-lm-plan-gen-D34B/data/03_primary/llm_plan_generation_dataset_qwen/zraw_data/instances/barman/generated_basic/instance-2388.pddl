(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot417 shot192 - shot
      ingredient140 ingredient238 ingredient361 - ingredient
      cocktail84 - cocktail
      dispenser50 dispenser440 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot417)
  (ontable shot192)
  (dispenses dispenser50 ingredient140)
  (dispenses dispenser440 ingredient238)
  (dispenses dispenser421 ingredient361)
  (clean shaker42)
  (clean shot417)
  (clean shot192)
  (empty shaker42)
  (empty shot417)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient140)
  (cocktail-part2 cocktail84 ingredient361)
)
 (:goal
  (and
      (contains shot417 cocktail84)
)))
