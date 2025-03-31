(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot435 shot352 - shot
      ingredient371 ingredient402 ingredient373 - ingredient
      cocktail152 - cocktail
      dispenser0 dispenser234 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot435)
  (ontable shot352)
  (dispenses dispenser0 ingredient371)
  (dispenses dispenser234 ingredient402)
  (dispenses dispenser26 ingredient373)
  (clean shaker207)
  (clean shot435)
  (clean shot352)
  (empty shaker207)
  (empty shot435)
  (empty shot352)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail152 ingredient371)
  (cocktail-part2 cocktail152 ingredient402)
)
 (:goal
  (and
      (contains shot435 cocktail152)
)))
