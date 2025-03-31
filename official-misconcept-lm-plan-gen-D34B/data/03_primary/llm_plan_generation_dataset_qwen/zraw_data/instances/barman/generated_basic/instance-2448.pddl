(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot211 shot430 shot161 - shot
      ingredient122 ingredient267 ingredient8 ingredient160 - ingredient
      cocktail48 - cocktail
      dispenser206 dispenser134 dispenser114 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot211)
  (ontable shot430)
  (ontable shot161)
  (dispenses dispenser206 ingredient122)
  (dispenses dispenser134 ingredient267)
  (dispenses dispenser114 ingredient8)
  (dispenses dispenser214 ingredient160)
  (clean shaker489)
  (clean shot211)
  (clean shot430)
  (clean shot161)
  (empty shaker489)
  (empty shot211)
  (empty shot430)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient267)
  (cocktail-part2 cocktail48 ingredient122)
)
 (:goal
  (and
      (contains shot211 cocktail48)
      (contains shot430 cocktail48)
)))
