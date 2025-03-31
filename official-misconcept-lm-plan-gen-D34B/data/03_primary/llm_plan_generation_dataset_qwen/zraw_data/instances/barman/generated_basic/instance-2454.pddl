(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot315 shot306 shot235 - shot
      ingredient36 ingredient400 ingredient412 ingredient307 - ingredient
      cocktail20 - cocktail
      dispenser462 dispenser128 dispenser461 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot315)
  (ontable shot306)
  (ontable shot235)
  (dispenses dispenser462 ingredient36)
  (dispenses dispenser128 ingredient400)
  (dispenses dispenser461 ingredient412)
  (dispenses dispenser478 ingredient307)
  (clean shaker449)
  (clean shot315)
  (clean shot306)
  (clean shot235)
  (empty shaker449)
  (empty shot315)
  (empty shot306)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail20 ingredient412)
  (cocktail-part2 cocktail20 ingredient307)
)
 (:goal
  (and
      (contains shot315 cocktail20)
      (contains shot306 ingredient412)
)))
