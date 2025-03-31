(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot410 shot70 - shot
      ingredient88 ingredient426 ingredient471 - ingredient
      cocktail254 - cocktail
      dispenser162 dispenser234 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot410)
  (ontable shot70)
  (dispenses dispenser162 ingredient88)
  (dispenses dispenser234 ingredient426)
  (dispenses dispenser246 ingredient471)
  (clean shaker416)
  (clean shot410)
  (clean shot70)
  (empty shaker416)
  (empty shot410)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail254 ingredient88)
  (cocktail-part2 cocktail254 ingredient471)
)
 (:goal
  (and
      (contains shot410 cocktail254)
)))
