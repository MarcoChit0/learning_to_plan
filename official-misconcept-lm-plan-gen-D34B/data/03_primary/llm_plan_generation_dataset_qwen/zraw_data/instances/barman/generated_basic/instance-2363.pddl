(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot444 - shot
      ingredient155 ingredient216 ingredient473 ingredient37 - ingredient
      cocktail383 - cocktail
      dispenser195 dispenser494 dispenser449 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot444)
  (dispenses dispenser195 ingredient155)
  (dispenses dispenser494 ingredient216)
  (dispenses dispenser449 ingredient473)
  (dispenses dispenser458 ingredient37)
  (clean shaker64)
  (clean shot444)
  (empty shaker64)
  (empty shot444)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail383 ingredient155)
  (cocktail-part2 cocktail383 ingredient216)
)
 (:goal
  (and
      (contains shot444 cocktail383)
)))
