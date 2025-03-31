(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot497 shot333 - shot
      ingredient112 ingredient493 - ingredient
      cocktail269 - cocktail
      dispenser45 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot497)
  (ontable shot333)
  (dispenses dispenser45 ingredient112)
  (dispenses dispenser88 ingredient493)
  (clean shaker186)
  (clean shot497)
  (clean shot333)
  (empty shaker186)
  (empty shot497)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient112)
  (cocktail-part2 cocktail269 ingredient493)
)
 (:goal
  (and
      (contains shot497 cocktail269)
)))
