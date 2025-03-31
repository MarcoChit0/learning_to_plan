(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot410 shot158 - shot
      ingredient46 ingredient398 ingredient313 - ingredient
      cocktail356 - cocktail
      dispenser343 dispenser167 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot410)
  (ontable shot158)
  (dispenses dispenser343 ingredient46)
  (dispenses dispenser167 ingredient398)
  (dispenses dispenser182 ingredient313)
  (clean shaker99)
  (clean shot410)
  (clean shot158)
  (empty shaker99)
  (empty shot410)
  (empty shot158)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient46)
  (cocktail-part2 cocktail356 ingredient313)
)
 (:goal
  (and
      (contains shot410 cocktail356)
)))
