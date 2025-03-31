(define (problem prob)
 (:domain barman)
 (:objects 
      shaker231 - shaker
      left right - hand
      shot71 shot218 shot358 - shot
      ingredient167 ingredient438 ingredient149 - ingredient
      cocktail480 - cocktail
      dispenser120 dispenser100 dispenser36 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker231)
  (ontable shot71)
  (ontable shot218)
  (ontable shot358)
  (dispenses dispenser120 ingredient167)
  (dispenses dispenser100 ingredient438)
  (dispenses dispenser36 ingredient149)
  (clean shaker231)
  (clean shot71)
  (clean shot218)
  (clean shot358)
  (empty shaker231)
  (empty shot71)
  (empty shot218)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker231 l0)
  (shaker-level shaker231 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient149)
  (cocktail-part2 cocktail480 ingredient167)
)
 (:goal
  (and
      (contains shot71 cocktail480)
      (contains shot218 cocktail480)
)))
