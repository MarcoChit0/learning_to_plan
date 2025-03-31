(define (problem prob)
 (:domain barman)
 (:objects 
      shaker492 - shaker
      left right - hand
      shot432 shot223 shot252 - shot
      ingredient292 ingredient179 ingredient270 - ingredient
      cocktail293 - cocktail
      dispenser139 dispenser153 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker492)
  (ontable shot432)
  (ontable shot223)
  (ontable shot252)
  (dispenses dispenser139 ingredient292)
  (dispenses dispenser153 ingredient179)
  (dispenses dispenser85 ingredient270)
  (clean shaker492)
  (clean shot432)
  (clean shot223)
  (clean shot252)
  (empty shaker492)
  (empty shot432)
  (empty shot223)
  (empty shot252)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker492 l0)
  (shaker-level shaker492 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient270)
  (cocktail-part2 cocktail293 ingredient179)
)
 (:goal
  (and
      (contains shot432 cocktail293)
      (contains shot223 cocktail293)
)))
