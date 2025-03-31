(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot69 shot191 - shot
      ingredient392 ingredient431 ingredient111 ingredient332 - ingredient
      cocktail291 - cocktail
      dispenser132 dispenser398 dispenser419 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot69)
  (ontable shot191)
  (dispenses dispenser132 ingredient392)
  (dispenses dispenser398 ingredient431)
  (dispenses dispenser419 ingredient111)
  (dispenses dispenser146 ingredient332)
  (clean shaker104)
  (clean shot69)
  (clean shot191)
  (empty shaker104)
  (empty shot69)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient431)
  (cocktail-part2 cocktail291 ingredient392)
)
 (:goal
  (and
      (contains shot69 cocktail291)
)))
