(define (problem prob)
 (:domain barman)
 (:objects 
      shaker346 - shaker
      left right - hand
      shot283 shot229 shot3 - shot
      ingredient383 ingredient396 - ingredient
      cocktail353 - cocktail
      dispenser259 dispenser164 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker346)
  (ontable shot283)
  (ontable shot229)
  (ontable shot3)
  (dispenses dispenser259 ingredient383)
  (dispenses dispenser164 ingredient396)
  (clean shaker346)
  (clean shot283)
  (clean shot229)
  (clean shot3)
  (empty shaker346)
  (empty shot283)
  (empty shot229)
  (empty shot3)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker346 l0)
  (shaker-level shaker346 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail353 ingredient383)
  (cocktail-part2 cocktail353 ingredient396)
)
 (:goal
  (and
      (contains shot283 cocktail353)
      (contains shot229 cocktail353)
)))
