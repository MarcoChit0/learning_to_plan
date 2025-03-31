(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot417 - shot
      ingredient354 ingredient399 ingredient84 - ingredient
      cocktail302 - cocktail
      dispenser150 dispenser480 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot417)
  (dispenses dispenser150 ingredient354)
  (dispenses dispenser480 ingredient399)
  (dispenses dispenser418 ingredient84)
  (clean shaker321)
  (clean shot417)
  (empty shaker321)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient399)
  (cocktail-part2 cocktail302 ingredient354)
)
 (:goal
  (and
      (contains shot417 cocktail302)
)))
