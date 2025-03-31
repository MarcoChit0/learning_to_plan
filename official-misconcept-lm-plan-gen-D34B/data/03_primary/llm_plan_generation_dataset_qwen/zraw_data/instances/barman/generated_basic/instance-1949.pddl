(define (problem prob)
 (:domain barman)
 (:objects 
      shaker40 - shaker
      left right - hand
      shot108 shot256 shot486 - shot
      ingredient12 ingredient23 ingredient384 ingredient19 - ingredient
      cocktail474 - cocktail
      dispenser488 dispenser50 dispenser422 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker40)
  (ontable shot108)
  (ontable shot256)
  (ontable shot486)
  (dispenses dispenser488 ingredient12)
  (dispenses dispenser50 ingredient23)
  (dispenses dispenser422 ingredient384)
  (dispenses dispenser492 ingredient19)
  (clean shaker40)
  (clean shot108)
  (clean shot256)
  (clean shot486)
  (empty shaker40)
  (empty shot108)
  (empty shot256)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker40 l0)
  (shaker-level shaker40 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail474 ingredient19)
  (cocktail-part2 cocktail474 ingredient12)
)
 (:goal
  (and
      (contains shot108 cocktail474)
      (contains shot256 cocktail474)
)))
