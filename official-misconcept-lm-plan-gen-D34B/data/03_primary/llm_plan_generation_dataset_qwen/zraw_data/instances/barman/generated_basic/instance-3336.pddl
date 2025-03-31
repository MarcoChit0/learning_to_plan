(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot207 shot209 - shot
      ingredient392 ingredient305 - ingredient
      cocktail468 - cocktail
      dispenser428 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot207)
  (ontable shot209)
  (dispenses dispenser428 ingredient392)
  (dispenses dispenser479 ingredient305)
  (clean shaker109)
  (clean shot207)
  (clean shot209)
  (empty shaker109)
  (empty shot207)
  (empty shot209)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient305)
  (cocktail-part2 cocktail468 ingredient392)
)
 (:goal
  (and
      (contains shot207 cocktail468)
)))
