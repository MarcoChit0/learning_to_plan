(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot206 shot281 shot343 - shot
      ingredient469 ingredient448 - ingredient
      cocktail71 - cocktail
      dispenser286 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot206)
  (ontable shot281)
  (ontable shot343)
  (dispenses dispenser286 ingredient469)
  (dispenses dispenser283 ingredient448)
  (clean shaker10)
  (clean shot206)
  (clean shot281)
  (clean shot343)
  (empty shaker10)
  (empty shot206)
  (empty shot281)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail71 ingredient448)
  (cocktail-part2 cocktail71 ingredient469)
)
 (:goal
  (and
      (contains shot206 cocktail71)
      (contains shot281 cocktail71)
)))
