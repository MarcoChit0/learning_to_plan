(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot52 shot418 - shot
      ingredient394 ingredient183 - ingredient
      cocktail1 - cocktail
      dispenser220 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot52)
  (ontable shot418)
  (dispenses dispenser220 ingredient394)
  (dispenses dispenser3 ingredient183)
  (clean shaker151)
  (clean shot52)
  (clean shot418)
  (empty shaker151)
  (empty shot52)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient394)
  (cocktail-part2 cocktail1 ingredient183)
)
 (:goal
  (and
      (contains shot52 cocktail1)
)))
