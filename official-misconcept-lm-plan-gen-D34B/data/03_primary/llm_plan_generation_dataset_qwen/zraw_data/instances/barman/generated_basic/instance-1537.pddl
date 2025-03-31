(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot310 shot28 - shot
      ingredient173 ingredient349 ingredient168 ingredient149 - ingredient
      cocktail309 - cocktail
      dispenser401 dispenser131 dispenser85 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot310)
  (ontable shot28)
  (dispenses dispenser401 ingredient173)
  (dispenses dispenser131 ingredient349)
  (dispenses dispenser85 ingredient168)
  (dispenses dispenser149 ingredient149)
  (clean shaker107)
  (clean shot310)
  (clean shot28)
  (empty shaker107)
  (empty shot310)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient168)
  (cocktail-part2 cocktail309 ingredient173)
)
 (:goal
  (and
      (contains shot310 cocktail309)
)))
