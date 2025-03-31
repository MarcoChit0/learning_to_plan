(define (problem prob)
 (:domain barman)
 (:objects 
      shaker392 - shaker
      left right - hand
      shot223 shot7 shot481 - shot
      ingredient243 ingredient357 ingredient162 ingredient380 - ingredient
      cocktail490 - cocktail
      dispenser398 dispenser49 dispenser44 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker392)
  (ontable shot223)
  (ontable shot7)
  (ontable shot481)
  (dispenses dispenser398 ingredient243)
  (dispenses dispenser49 ingredient357)
  (dispenses dispenser44 ingredient162)
  (dispenses dispenser32 ingredient380)
  (clean shaker392)
  (clean shot223)
  (clean shot7)
  (clean shot481)
  (empty shaker392)
  (empty shot223)
  (empty shot7)
  (empty shot481)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker392 l0)
  (shaker-level shaker392 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail490 ingredient162)
  (cocktail-part2 cocktail490 ingredient357)
)
 (:goal
  (and
      (contains shot223 cocktail490)
      (contains shot7 ingredient162)
)))
