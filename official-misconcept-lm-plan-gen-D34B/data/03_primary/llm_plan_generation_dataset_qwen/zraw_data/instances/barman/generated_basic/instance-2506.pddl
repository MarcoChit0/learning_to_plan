(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot267 - shot
      ingredient371 ingredient499 - ingredient
      cocktail292 - cocktail
      dispenser94 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot267)
  (dispenses dispenser94 ingredient371)
  (dispenses dispenser6 ingredient499)
  (clean shaker64)
  (clean shot267)
  (empty shaker64)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient371)
  (cocktail-part2 cocktail292 ingredient499)
)
 (:goal
  (and
      (contains shot267 cocktail292)
)))
