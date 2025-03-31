(define (problem prob)
 (:domain barman)
 (:objects 
      shaker473 - shaker
      left right - hand
      shot440 shot302 - shot
      ingredient379 ingredient306 ingredient126 ingredient93 - ingredient
      cocktail380 - cocktail
      dispenser247 dispenser292 dispenser116 dispenser379 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker473)
  (ontable shot440)
  (ontable shot302)
  (dispenses dispenser247 ingredient379)
  (dispenses dispenser292 ingredient306)
  (dispenses dispenser116 ingredient126)
  (dispenses dispenser379 ingredient93)
  (clean shaker473)
  (clean shot440)
  (clean shot302)
  (empty shaker473)
  (empty shot440)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker473 l0)
  (shaker-level shaker473 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient306)
  (cocktail-part2 cocktail380 ingredient379)
)
 (:goal
  (and
      (contains shot440 cocktail380)
)))
