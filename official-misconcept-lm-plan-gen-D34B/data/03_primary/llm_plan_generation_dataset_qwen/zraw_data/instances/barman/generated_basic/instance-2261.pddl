(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot417 shot194 - shot
      ingredient365 ingredient43 ingredient75 ingredient407 - ingredient
      cocktail314 - cocktail
      dispenser466 dispenser141 dispenser66 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot417)
  (ontable shot194)
  (dispenses dispenser466 ingredient365)
  (dispenses dispenser141 ingredient43)
  (dispenses dispenser66 ingredient75)
  (dispenses dispenser474 ingredient407)
  (clean shaker425)
  (clean shot417)
  (clean shot194)
  (empty shaker425)
  (empty shot417)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail314 ingredient43)
  (cocktail-part2 cocktail314 ingredient365)
)
 (:goal
  (and
      (contains shot417 cocktail314)
)))
