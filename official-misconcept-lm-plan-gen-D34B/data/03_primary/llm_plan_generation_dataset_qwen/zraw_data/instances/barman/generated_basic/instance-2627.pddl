(define (problem prob)
 (:domain barman)
 (:objects 
      shaker254 - shaker
      left right - hand
      shot287 shot68 - shot
      ingredient84 ingredient256 ingredient153 - ingredient
      cocktail127 - cocktail
      dispenser343 dispenser270 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker254)
  (ontable shot287)
  (ontable shot68)
  (dispenses dispenser343 ingredient84)
  (dispenses dispenser270 ingredient256)
  (dispenses dispenser399 ingredient153)
  (clean shaker254)
  (clean shot287)
  (clean shot68)
  (empty shaker254)
  (empty shot287)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker254 l0)
  (shaker-level shaker254 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient84)
  (cocktail-part2 cocktail127 ingredient153)
)
 (:goal
  (and
      (contains shot287 cocktail127)
)))
