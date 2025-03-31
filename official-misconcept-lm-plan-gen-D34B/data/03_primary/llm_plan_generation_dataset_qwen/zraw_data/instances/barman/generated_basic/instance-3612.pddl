(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot45 shot20 shot108 - shot
      ingredient27 ingredient151 ingredient310 - ingredient
      cocktail88 - cocktail
      dispenser459 dispenser104 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot45)
  (ontable shot20)
  (ontable shot108)
  (dispenses dispenser459 ingredient27)
  (dispenses dispenser104 ingredient151)
  (dispenses dispenser275 ingredient310)
  (clean shaker67)
  (clean shot45)
  (clean shot20)
  (clean shot108)
  (empty shaker67)
  (empty shot45)
  (empty shot20)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient310)
  (cocktail-part2 cocktail88 ingredient151)
)
 (:goal
  (and
      (contains shot45 cocktail88)
      (contains shot20 ingredient27)
)))
