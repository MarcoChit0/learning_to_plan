(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot303 shot373 - shot
      ingredient462 ingredient254 ingredient474 - ingredient
      cocktail384 - cocktail
      dispenser400 dispenser156 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot303)
  (ontable shot373)
  (dispenses dispenser400 ingredient462)
  (dispenses dispenser156 ingredient254)
  (dispenses dispenser71 ingredient474)
  (clean shaker62)
  (clean shot303)
  (clean shot373)
  (empty shaker62)
  (empty shot303)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient474)
  (cocktail-part2 cocktail384 ingredient254)
)
 (:goal
  (and
      (contains shot303 cocktail384)
)))
