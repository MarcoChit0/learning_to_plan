(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot206 - shot
      ingredient59 ingredient345 - ingredient
      cocktail20 - cocktail
      dispenser149 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot206)
  (dispenses dispenser149 ingredient59)
  (dispenses dispenser377 ingredient345)
  (clean shaker64)
  (clean shot206)
  (empty shaker64)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail20 ingredient345)
  (cocktail-part2 cocktail20 ingredient59)
)
 (:goal
  (and
      (contains shot206 cocktail20)
)))
