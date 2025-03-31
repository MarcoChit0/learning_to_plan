(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot105 shot343 shot68 - shot
      ingredient116 ingredient348 ingredient468 ingredient300 - ingredient
      cocktail95 - cocktail
      dispenser267 dispenser331 dispenser309 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot105)
  (ontable shot343)
  (ontable shot68)
  (dispenses dispenser267 ingredient116)
  (dispenses dispenser331 ingredient348)
  (dispenses dispenser309 ingredient468)
  (dispenses dispenser478 ingredient300)
  (clean shaker104)
  (clean shot105)
  (clean shot343)
  (clean shot68)
  (empty shaker104)
  (empty shot105)
  (empty shot343)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient468)
  (cocktail-part2 cocktail95 ingredient348)
)
 (:goal
  (and
      (contains shot105 cocktail95)
      (contains shot343 ingredient468)
)))
