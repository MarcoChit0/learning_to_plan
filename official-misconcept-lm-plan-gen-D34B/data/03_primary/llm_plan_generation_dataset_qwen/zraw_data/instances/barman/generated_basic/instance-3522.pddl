(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot379 shot86 - shot
      ingredient63 ingredient17 ingredient95 ingredient451 - ingredient
      cocktail418 - cocktail
      dispenser145 dispenser384 dispenser72 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot379)
  (ontable shot86)
  (dispenses dispenser145 ingredient63)
  (dispenses dispenser384 ingredient17)
  (dispenses dispenser72 ingredient95)
  (dispenses dispenser436 ingredient451)
  (clean shaker64)
  (clean shot379)
  (clean shot86)
  (empty shaker64)
  (empty shot379)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient451)
  (cocktail-part2 cocktail418 ingredient17)
)
 (:goal
  (and
      (contains shot379 cocktail418)
)))
