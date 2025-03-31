(define (problem prob)
 (:domain barman)
 (:objects 
      shaker223 - shaker
      left right - hand
      shot478 shot146 - shot
      ingredient70 ingredient330 ingredient481 ingredient254 - ingredient
      cocktail178 - cocktail
      dispenser308 dispenser442 dispenser353 dispenser37 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker223)
  (ontable shot478)
  (ontable shot146)
  (dispenses dispenser308 ingredient70)
  (dispenses dispenser442 ingredient330)
  (dispenses dispenser353 ingredient481)
  (dispenses dispenser37 ingredient254)
  (clean shaker223)
  (clean shot478)
  (clean shot146)
  (empty shaker223)
  (empty shot478)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker223 l0)
  (shaker-level shaker223 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient330)
  (cocktail-part2 cocktail178 ingredient70)
)
 (:goal
  (and
      (contains shot478 cocktail178)
)))
