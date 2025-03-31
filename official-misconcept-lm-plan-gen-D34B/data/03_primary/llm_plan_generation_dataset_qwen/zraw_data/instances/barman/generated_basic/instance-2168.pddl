(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot320 shot359 - shot
      ingredient122 ingredient100 - ingredient
      cocktail131 - cocktail
      dispenser13 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot320)
  (ontable shot359)
  (dispenses dispenser13 ingredient122)
  (dispenses dispenser234 ingredient100)
  (clean shaker340)
  (clean shot320)
  (clean shot359)
  (empty shaker340)
  (empty shot320)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient100)
  (cocktail-part2 cocktail131 ingredient122)
)
 (:goal
  (and
      (contains shot320 cocktail131)
)))
