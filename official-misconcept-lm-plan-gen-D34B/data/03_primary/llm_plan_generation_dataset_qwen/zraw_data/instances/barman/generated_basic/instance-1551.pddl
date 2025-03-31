(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot327 - shot
      ingredient265 ingredient441 - ingredient
      cocktail230 - cocktail
      dispenser86 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot327)
  (dispenses dispenser86 ingredient265)
  (dispenses dispenser420 ingredient441)
  (clean shaker64)
  (clean shot327)
  (empty shaker64)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail230 ingredient265)
  (cocktail-part2 cocktail230 ingredient441)
)
 (:goal
  (and
      (contains shot327 cocktail230)
)))
