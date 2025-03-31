(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot292 shot152 shot451 shot114 - shot
      ingredient343 ingredient34 - ingredient
      cocktail77 - cocktail
      dispenser389 dispenser67 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot292)
  (ontable shot152)
  (ontable shot451)
  (ontable shot114)
  (dispenses dispenser389 ingredient343)
  (dispenses dispenser67 ingredient34)
  (clean shaker443)
  (clean shot292)
  (clean shot152)
  (clean shot451)
  (clean shot114)
  (empty shaker443)
  (empty shot292)
  (empty shot152)
  (empty shot451)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail77 ingredient34)
  (cocktail-part2 cocktail77 ingredient343)
)
 (:goal
  (and
      (contains shot292 cocktail77)
      (contains shot152 ingredient343)
      (contains shot451 cocktail77)
)))
