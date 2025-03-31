(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot162 shot150 - shot
      ingredient453 ingredient20 ingredient270 - ingredient
      cocktail13 - cocktail
      dispenser315 dispenser1 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot162)
  (ontable shot150)
  (dispenses dispenser315 ingredient453)
  (dispenses dispenser1 ingredient20)
  (dispenses dispenser291 ingredient270)
  (clean shaker64)
  (clean shot162)
  (clean shot150)
  (empty shaker64)
  (empty shot162)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient453)
  (cocktail-part2 cocktail13 ingredient270)
)
 (:goal
  (and
      (contains shot162 cocktail13)
)))
