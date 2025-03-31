(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot325 - shot
      ingredient307 ingredient36 ingredient237 ingredient420 - ingredient
      cocktail102 - cocktail
      dispenser250 dispenser427 dispenser266 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot325)
  (dispenses dispenser250 ingredient307)
  (dispenses dispenser427 ingredient36)
  (dispenses dispenser266 ingredient237)
  (dispenses dispenser410 ingredient420)
  (clean shaker151)
  (clean shot325)
  (empty shaker151)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail102 ingredient307)
  (cocktail-part2 cocktail102 ingredient237)
)
 (:goal
  (and
      (contains shot325 cocktail102)
)))
