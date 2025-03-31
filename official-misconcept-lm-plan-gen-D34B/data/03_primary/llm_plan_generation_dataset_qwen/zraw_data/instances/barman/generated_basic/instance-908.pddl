(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot380 shot197 - shot
      ingredient317 ingredient273 - ingredient
      cocktail1 - cocktail
      dispenser227 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot380)
  (ontable shot197)
  (dispenses dispenser227 ingredient317)
  (dispenses dispenser492 ingredient273)
  (clean shaker43)
  (clean shot380)
  (clean shot197)
  (empty shaker43)
  (empty shot380)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient273)
  (cocktail-part2 cocktail1 ingredient317)
)
 (:goal
  (and
      (contains shot380 cocktail1)
)))
