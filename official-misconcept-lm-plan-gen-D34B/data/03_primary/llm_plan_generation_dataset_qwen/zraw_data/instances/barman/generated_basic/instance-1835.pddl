(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot293 shot21 - shot
      ingredient485 ingredient58 ingredient293 - ingredient
      cocktail355 - cocktail
      dispenser362 dispenser375 dispenser164 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot293)
  (ontable shot21)
  (dispenses dispenser362 ingredient485)
  (dispenses dispenser375 ingredient58)
  (dispenses dispenser164 ingredient293)
  (clean shaker64)
  (clean shot293)
  (clean shot21)
  (empty shaker64)
  (empty shot293)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient293)
  (cocktail-part2 cocktail355 ingredient58)
)
 (:goal
  (and
      (contains shot293 cocktail355)
)))
