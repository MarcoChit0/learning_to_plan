(define (problem prob)
 (:domain barman)
 (:objects 
      shaker127 - shaker
      left right - hand
      shot26 shot15 - shot
      ingredient257 ingredient196 ingredient232 ingredient282 - ingredient
      cocktail19 - cocktail
      dispenser380 dispenser311 dispenser459 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker127)
  (ontable shot26)
  (ontable shot15)
  (dispenses dispenser380 ingredient257)
  (dispenses dispenser311 ingredient196)
  (dispenses dispenser459 ingredient232)
  (dispenses dispenser255 ingredient282)
  (clean shaker127)
  (clean shot26)
  (clean shot15)
  (empty shaker127)
  (empty shot26)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker127 l0)
  (shaker-level shaker127 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail19 ingredient257)
  (cocktail-part2 cocktail19 ingredient232)
)
 (:goal
  (and
      (contains shot26 cocktail19)
)))
