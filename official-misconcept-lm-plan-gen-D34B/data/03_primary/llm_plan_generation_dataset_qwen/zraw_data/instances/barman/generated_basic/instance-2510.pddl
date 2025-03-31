(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot85 - shot
      ingredient456 ingredient493 - ingredient
      cocktail30 - cocktail
      dispenser118 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot85)
  (dispenses dispenser118 ingredient456)
  (dispenses dispenser128 ingredient493)
  (clean shaker57)
  (clean shot85)
  (empty shaker57)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient456)
  (cocktail-part2 cocktail30 ingredient493)
)
 (:goal
  (and
      (contains shot85 cocktail30)
)))
