(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot223 shot193 shot420 - shot
      ingredient299 ingredient257 ingredient404 ingredient245 - ingredient
      cocktail30 - cocktail
      dispenser329 dispenser349 dispenser38 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot223)
  (ontable shot193)
  (ontable shot420)
  (dispenses dispenser329 ingredient299)
  (dispenses dispenser349 ingredient257)
  (dispenses dispenser38 ingredient404)
  (dispenses dispenser240 ingredient245)
  (clean shaker340)
  (clean shot223)
  (clean shot193)
  (clean shot420)
  (empty shaker340)
  (empty shot223)
  (empty shot193)
  (empty shot420)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient245)
  (cocktail-part2 cocktail30 ingredient404)
)
 (:goal
  (and
      (contains shot223 cocktail30)
      (contains shot193 ingredient299)
)))
