(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot122 - shot
      ingredient30 ingredient212 ingredient450 - ingredient
      cocktail413 - cocktail
      dispenser225 dispenser341 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot122)
  (dispenses dispenser225 ingredient30)
  (dispenses dispenser341 ingredient212)
  (dispenses dispenser498 ingredient450)
  (clean shaker114)
  (clean shot122)
  (empty shaker114)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail413 ingredient30)
  (cocktail-part2 cocktail413 ingredient212)
)
 (:goal
  (and
      (contains shot122 cocktail413)
)))
