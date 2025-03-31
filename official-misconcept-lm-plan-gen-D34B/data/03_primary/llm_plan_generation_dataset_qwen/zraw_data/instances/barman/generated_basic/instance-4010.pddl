(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot419 shot215 shot374 - shot
      ingredient66 ingredient428 - ingredient
      cocktail75 - cocktail
      dispenser110 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot419)
  (ontable shot215)
  (ontable shot374)
  (dispenses dispenser110 ingredient66)
  (dispenses dispenser126 ingredient428)
  (clean shaker64)
  (clean shot419)
  (clean shot215)
  (clean shot374)
  (empty shaker64)
  (empty shot419)
  (empty shot215)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient428)
  (cocktail-part2 cocktail75 ingredient66)
)
 (:goal
  (and
      (contains shot419 cocktail75)
      (contains shot215 cocktail75)
)))
