(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot280 shot59 shot242 - shot
      ingredient168 ingredient299 ingredient470 ingredient426 - ingredient
      cocktail457 - cocktail
      dispenser70 dispenser98 dispenser468 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot280)
  (ontable shot59)
  (ontable shot242)
  (dispenses dispenser70 ingredient168)
  (dispenses dispenser98 ingredient299)
  (dispenses dispenser468 ingredient470)
  (dispenses dispenser115 ingredient426)
  (clean shaker377)
  (clean shot280)
  (clean shot59)
  (clean shot242)
  (empty shaker377)
  (empty shot280)
  (empty shot59)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient299)
  (cocktail-part2 cocktail457 ingredient426)
)
 (:goal
  (and
      (contains shot280 cocktail457)
      (contains shot59 ingredient426)
)))
