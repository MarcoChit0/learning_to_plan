(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot35 shot461 shot146 - shot
      ingredient205 ingredient284 ingredient438 ingredient453 - ingredient
      cocktail70 - cocktail
      dispenser455 dispenser270 dispenser200 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot35)
  (ontable shot461)
  (ontable shot146)
  (dispenses dispenser455 ingredient205)
  (dispenses dispenser270 ingredient284)
  (dispenses dispenser200 ingredient438)
  (dispenses dispenser266 ingredient453)
  (clean shaker34)
  (clean shot35)
  (clean shot461)
  (clean shot146)
  (empty shaker34)
  (empty shot35)
  (empty shot461)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient205)
  (cocktail-part2 cocktail70 ingredient438)
)
 (:goal
  (and
      (contains shot35 cocktail70)
      (contains shot461 ingredient438)
)))
