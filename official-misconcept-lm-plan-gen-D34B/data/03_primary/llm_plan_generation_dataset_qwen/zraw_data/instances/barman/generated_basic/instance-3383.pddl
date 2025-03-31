(define (problem prob)
 (:domain barman)
 (:objects 
      shaker471 - shaker
      left right - hand
      shot172 shot108 shot251 - shot
      ingredient153 ingredient83 ingredient397 - ingredient
      cocktail22 - cocktail
      dispenser26 dispenser277 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker471)
  (ontable shot172)
  (ontable shot108)
  (ontable shot251)
  (dispenses dispenser26 ingredient153)
  (dispenses dispenser277 ingredient83)
  (dispenses dispenser387 ingredient397)
  (clean shaker471)
  (clean shot172)
  (clean shot108)
  (clean shot251)
  (empty shaker471)
  (empty shot172)
  (empty shot108)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker471 l0)
  (shaker-level shaker471 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient83)
  (cocktail-part2 cocktail22 ingredient153)
)
 (:goal
  (and
      (contains shot172 cocktail22)
      (contains shot108 cocktail22)
)))
