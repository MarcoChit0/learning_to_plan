(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot107 shot274 - shot
      ingredient314 ingredient141 ingredient407 ingredient186 - ingredient
      cocktail14 - cocktail
      dispenser493 dispenser353 dispenser151 dispenser105 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot107)
  (ontable shot274)
  (dispenses dispenser493 ingredient314)
  (dispenses dispenser353 ingredient141)
  (dispenses dispenser151 ingredient407)
  (dispenses dispenser105 ingredient186)
  (clean shaker94)
  (clean shot107)
  (clean shot274)
  (empty shaker94)
  (empty shot107)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient314)
  (cocktail-part2 cocktail14 ingredient186)
)
 (:goal
  (and
      (contains shot107 cocktail14)
)))
