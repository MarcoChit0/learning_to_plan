(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot69 shot153 - shot
      ingredient170 ingredient123 ingredient49 ingredient71 - ingredient
      cocktail430 - cocktail
      dispenser270 dispenser283 dispenser46 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot69)
  (ontable shot153)
  (dispenses dispenser270 ingredient170)
  (dispenses dispenser283 ingredient123)
  (dispenses dispenser46 ingredient49)
  (dispenses dispenser252 ingredient71)
  (clean shaker64)
  (clean shot69)
  (clean shot153)
  (empty shaker64)
  (empty shot69)
  (empty shot153)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient123)
  (cocktail-part2 cocktail430 ingredient170)
)
 (:goal
  (and
      (contains shot69 cocktail430)
)))
