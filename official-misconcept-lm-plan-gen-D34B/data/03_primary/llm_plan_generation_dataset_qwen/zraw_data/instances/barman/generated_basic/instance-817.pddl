(define (problem prob)
 (:domain barman)
 (:objects 
      shaker263 - shaker
      left right - hand
      shot122 shot112 - shot
      ingredient250 ingredient141 ingredient128 ingredient290 - ingredient
      cocktail1 - cocktail
      dispenser309 dispenser102 dispenser57 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker263)
  (ontable shot122)
  (ontable shot112)
  (dispenses dispenser309 ingredient250)
  (dispenses dispenser102 ingredient141)
  (dispenses dispenser57 ingredient128)
  (dispenses dispenser337 ingredient290)
  (clean shaker263)
  (clean shot122)
  (clean shot112)
  (empty shaker263)
  (empty shot122)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker263 l0)
  (shaker-level shaker263 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient128)
  (cocktail-part2 cocktail1 ingredient250)
)
 (:goal
  (and
      (contains shot122 cocktail1)
)))
