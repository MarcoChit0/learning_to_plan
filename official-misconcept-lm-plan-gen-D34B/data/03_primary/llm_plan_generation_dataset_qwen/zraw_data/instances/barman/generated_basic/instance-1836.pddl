(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot105 shot381 - shot
      ingredient230 ingredient201 ingredient105 - ingredient
      cocktail240 - cocktail
      dispenser78 dispenser458 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot105)
  (ontable shot381)
  (dispenses dispenser78 ingredient230)
  (dispenses dispenser458 ingredient201)
  (dispenses dispenser225 ingredient105)
  (clean shaker116)
  (clean shot105)
  (clean shot381)
  (empty shaker116)
  (empty shot105)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient230)
  (cocktail-part2 cocktail240 ingredient201)
)
 (:goal
  (and
      (contains shot105 cocktail240)
)))
