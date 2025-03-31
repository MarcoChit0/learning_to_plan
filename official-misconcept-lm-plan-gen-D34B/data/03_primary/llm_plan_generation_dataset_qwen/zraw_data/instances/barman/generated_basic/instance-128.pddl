(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot188 shot83 - shot
      ingredient219 ingredient459 - ingredient
      cocktail1 - cocktail
      dispenser276 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot188)
  (ontable shot83)
  (dispenses dispenser276 ingredient219)
  (dispenses dispenser364 ingredient459)
  (clean shaker90)
  (clean shot188)
  (clean shot83)
  (empty shaker90)
  (empty shot188)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient219)
  (cocktail-part2 cocktail1 ingredient459)
)
 (:goal
  (and
      (contains shot188 cocktail1)
)))
