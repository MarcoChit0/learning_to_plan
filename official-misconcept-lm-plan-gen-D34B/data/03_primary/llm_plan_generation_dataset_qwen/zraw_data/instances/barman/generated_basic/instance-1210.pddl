(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot295 shot331 shot212 - shot
      ingredient287 ingredient205 ingredient467 - ingredient
      cocktail483 - cocktail
      dispenser38 dispenser100 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot295)
  (ontable shot331)
  (ontable shot212)
  (dispenses dispenser38 ingredient287)
  (dispenses dispenser100 ingredient205)
  (dispenses dispenser474 ingredient467)
  (clean shaker72)
  (clean shot295)
  (clean shot331)
  (clean shot212)
  (empty shaker72)
  (empty shot295)
  (empty shot331)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail483 ingredient467)
  (cocktail-part2 cocktail483 ingredient205)
)
 (:goal
  (and
      (contains shot295 cocktail483)
      (contains shot331 cocktail483)
)))
