(define (problem prob)
 (:domain barman)
 (:objects 
      shaker145 - shaker
      left right - hand
      shot449 shot250 - shot
      ingredient167 ingredient140 ingredient228 - ingredient
      cocktail347 - cocktail
      dispenser149 dispenser416 dispenser174 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker145)
  (ontable shot449)
  (ontable shot250)
  (dispenses dispenser149 ingredient167)
  (dispenses dispenser416 ingredient140)
  (dispenses dispenser174 ingredient228)
  (clean shaker145)
  (clean shot449)
  (clean shot250)
  (empty shaker145)
  (empty shot449)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker145 l0)
  (shaker-level shaker145 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient228)
  (cocktail-part2 cocktail347 ingredient167)
)
 (:goal
  (and
      (contains shot449 cocktail347)
)))
