(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot71 shot143 - shot
      ingredient92 ingredient59 - ingredient
      cocktail131 - cocktail
      dispenser21 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot71)
  (ontable shot143)
  (dispenses dispenser21 ingredient92)
  (dispenses dispenser130 ingredient59)
  (clean shaker307)
  (clean shot71)
  (clean shot143)
  (empty shaker307)
  (empty shot71)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient59)
  (cocktail-part2 cocktail131 ingredient92)
)
 (:goal
  (and
      (contains shot71 cocktail131)
)))
