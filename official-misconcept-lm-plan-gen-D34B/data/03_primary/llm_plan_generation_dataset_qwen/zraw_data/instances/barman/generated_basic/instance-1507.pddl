(define (problem prob)
 (:domain barman)
 (:objects 
      shaker309 - shaker
      left right - hand
      shot38 shot299 - shot
      ingredient38 ingredient306 ingredient154 ingredient222 - ingredient
      cocktail63 - cocktail
      dispenser291 dispenser182 dispenser107 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker309)
  (ontable shot38)
  (ontable shot299)
  (dispenses dispenser291 ingredient38)
  (dispenses dispenser182 ingredient306)
  (dispenses dispenser107 ingredient154)
  (dispenses dispenser368 ingredient222)
  (clean shaker309)
  (clean shot38)
  (clean shot299)
  (empty shaker309)
  (empty shot38)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker309 l0)
  (shaker-level shaker309 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient222)
  (cocktail-part2 cocktail63 ingredient306)
)
 (:goal
  (and
      (contains shot38 cocktail63)
)))
