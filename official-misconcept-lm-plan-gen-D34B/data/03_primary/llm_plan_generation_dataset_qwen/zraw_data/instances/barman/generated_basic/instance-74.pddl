(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot232 shot499 - shot
      ingredient4 ingredient443 ingredient490 - ingredient
      cocktail1 - cocktail
      dispenser291 dispenser34 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot232)
  (ontable shot499)
  (dispenses dispenser291 ingredient4)
  (dispenses dispenser34 ingredient443)
  (dispenses dispenser278 ingredient490)
  (clean shaker211)
  (clean shot232)
  (clean shot499)
  (empty shaker211)
  (empty shot232)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient443)
  (cocktail-part2 cocktail1 ingredient490)
)
 (:goal
  (and
      (contains shot232 cocktail1)
)))
