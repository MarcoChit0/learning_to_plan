(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot14 shot59 - shot
      ingredient30 ingredient399 ingredient102 ingredient316 - ingredient
      cocktail410 - cocktail
      dispenser406 dispenser256 dispenser322 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot14)
  (ontable shot59)
  (dispenses dispenser406 ingredient30)
  (dispenses dispenser256 ingredient399)
  (dispenses dispenser322 ingredient102)
  (dispenses dispenser357 ingredient316)
  (clean shaker474)
  (clean shot14)
  (clean shot59)
  (empty shaker474)
  (empty shot14)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail410 ingredient102)
  (cocktail-part2 cocktail410 ingredient399)
)
 (:goal
  (and
      (contains shot14 cocktail410)
)))
