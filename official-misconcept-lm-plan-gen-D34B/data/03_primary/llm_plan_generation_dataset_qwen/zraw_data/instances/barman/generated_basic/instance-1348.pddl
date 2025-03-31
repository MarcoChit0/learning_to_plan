(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot156 shot413 shot32 - shot
      ingredient41 ingredient21 ingredient228 - ingredient
      cocktail331 - cocktail
      dispenser40 dispenser82 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot156)
  (ontable shot413)
  (ontable shot32)
  (dispenses dispenser40 ingredient41)
  (dispenses dispenser82 ingredient21)
  (dispenses dispenser166 ingredient228)
  (clean shaker314)
  (clean shot156)
  (clean shot413)
  (clean shot32)
  (empty shaker314)
  (empty shot156)
  (empty shot413)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient41)
  (cocktail-part2 cocktail331 ingredient21)
)
 (:goal
  (and
      (contains shot156 cocktail331)
      (contains shot413 ingredient228)
)))
