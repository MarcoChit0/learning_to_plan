(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot401 shot259 - shot
      ingredient151 ingredient298 ingredient182 ingredient443 - ingredient
      cocktail412 - cocktail
      dispenser258 dispenser17 dispenser137 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot401)
  (ontable shot259)
  (dispenses dispenser258 ingredient151)
  (dispenses dispenser17 ingredient298)
  (dispenses dispenser137 ingredient182)
  (dispenses dispenser273 ingredient443)
  (clean shaker351)
  (clean shot401)
  (clean shot259)
  (empty shaker351)
  (empty shot401)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient182)
  (cocktail-part2 cocktail412 ingredient443)
)
 (:goal
  (and
      (contains shot401 cocktail412)
)))
