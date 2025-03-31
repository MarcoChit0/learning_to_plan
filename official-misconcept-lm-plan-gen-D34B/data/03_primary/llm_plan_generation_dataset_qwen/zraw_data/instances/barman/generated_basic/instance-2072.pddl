(define (problem prob)
 (:domain barman)
 (:objects 
      shaker302 - shaker
      left right - hand
      shot190 shot96 - shot
      ingredient410 ingredient124 - ingredient
      cocktail295 - cocktail
      dispenser130 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker302)
  (ontable shot190)
  (ontable shot96)
  (dispenses dispenser130 ingredient410)
  (dispenses dispenser223 ingredient124)
  (clean shaker302)
  (clean shot190)
  (clean shot96)
  (empty shaker302)
  (empty shot190)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker302 l0)
  (shaker-level shaker302 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail295 ingredient124)
  (cocktail-part2 cocktail295 ingredient410)
)
 (:goal
  (and
      (contains shot190 cocktail295)
)))
