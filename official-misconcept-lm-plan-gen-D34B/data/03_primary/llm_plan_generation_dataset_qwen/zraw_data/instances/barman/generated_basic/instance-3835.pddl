(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot74 shot465 shot328 - shot
      ingredient176 ingredient95 - ingredient
      cocktail155 - cocktail
      dispenser215 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot74)
  (ontable shot465)
  (ontable shot328)
  (dispenses dispenser215 ingredient176)
  (dispenses dispenser447 ingredient95)
  (clean shaker276)
  (clean shot74)
  (clean shot465)
  (clean shot328)
  (empty shaker276)
  (empty shot74)
  (empty shot465)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient95)
  (cocktail-part2 cocktail155 ingredient176)
)
 (:goal
  (and
      (contains shot74 cocktail155)
      (contains shot465 cocktail155)
)))
