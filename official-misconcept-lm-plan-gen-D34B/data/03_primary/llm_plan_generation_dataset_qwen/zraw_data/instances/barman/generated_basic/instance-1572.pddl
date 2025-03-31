(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot34 shot475 shot479 - shot
      ingredient471 ingredient281 ingredient469 ingredient324 - ingredient
      cocktail357 - cocktail
      dispenser160 dispenser38 dispenser138 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot34)
  (ontable shot475)
  (ontable shot479)
  (dispenses dispenser160 ingredient471)
  (dispenses dispenser38 ingredient281)
  (dispenses dispenser138 ingredient469)
  (dispenses dispenser412 ingredient324)
  (clean shaker307)
  (clean shot34)
  (clean shot475)
  (clean shot479)
  (empty shaker307)
  (empty shot34)
  (empty shot475)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient281)
  (cocktail-part2 cocktail357 ingredient471)
)
 (:goal
  (and
      (contains shot34 cocktail357)
      (contains shot475 cocktail357)
)))
