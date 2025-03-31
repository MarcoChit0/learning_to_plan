(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot120 shot221 - shot
      ingredient424 ingredient235 ingredient172 ingredient140 - ingredient
      cocktail1 - cocktail
      dispenser126 dispenser399 dispenser393 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot120)
  (ontable shot221)
  (dispenses dispenser126 ingredient424)
  (dispenses dispenser399 ingredient235)
  (dispenses dispenser393 ingredient172)
  (dispenses dispenser222 ingredient140)
  (clean shaker98)
  (clean shot120)
  (clean shot221)
  (empty shaker98)
  (empty shot120)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient140)
  (cocktail-part2 cocktail1 ingredient424)
)
 (:goal
  (and
      (contains shot120 cocktail1)
)))
