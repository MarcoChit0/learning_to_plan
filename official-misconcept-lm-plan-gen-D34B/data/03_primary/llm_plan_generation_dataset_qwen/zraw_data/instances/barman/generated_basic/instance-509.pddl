(define (problem prob)
 (:domain barman)
 (:objects 
      shaker208 - shaker
      left right - hand
      shot87 shot459 - shot
      ingredient266 ingredient354 ingredient233 ingredient337 - ingredient
      cocktail1 - cocktail
      dispenser258 dispenser77 dispenser290 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker208)
  (ontable shot87)
  (ontable shot459)
  (dispenses dispenser258 ingredient266)
  (dispenses dispenser77 ingredient354)
  (dispenses dispenser290 ingredient233)
  (dispenses dispenser180 ingredient337)
  (clean shaker208)
  (clean shot87)
  (clean shot459)
  (empty shaker208)
  (empty shot87)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker208 l0)
  (shaker-level shaker208 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient266)
  (cocktail-part2 cocktail1 ingredient233)
)
 (:goal
  (and
      (contains shot87 cocktail1)
)))
