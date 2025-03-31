(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot157 shot284 - shot
      ingredient99 ingredient370 ingredient300 - ingredient
      cocktail85 - cocktail
      dispenser100 dispenser472 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot157)
  (ontable shot284)
  (dispenses dispenser100 ingredient99)
  (dispenses dispenser472 ingredient370)
  (dispenses dispenser246 ingredient300)
  (clean shaker126)
  (clean shot157)
  (clean shot284)
  (empty shaker126)
  (empty shot157)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient300)
  (cocktail-part2 cocktail85 ingredient370)
)
 (:goal
  (and
      (contains shot157 cocktail85)
)))
