(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot39 - shot
      ingredient428 ingredient393 ingredient429 ingredient68 - ingredient
      cocktail282 - cocktail
      dispenser156 dispenser115 dispenser249 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot39)
  (dispenses dispenser156 ingredient428)
  (dispenses dispenser115 ingredient393)
  (dispenses dispenser249 ingredient429)
  (dispenses dispenser480 ingredient68)
  (clean shaker472)
  (clean shot39)
  (empty shaker472)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail282 ingredient429)
  (cocktail-part2 cocktail282 ingredient393)
)
 (:goal
  (and
      (contains shot39 cocktail282)
)))
