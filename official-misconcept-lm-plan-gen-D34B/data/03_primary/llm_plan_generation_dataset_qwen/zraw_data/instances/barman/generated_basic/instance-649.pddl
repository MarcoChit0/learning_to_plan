(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot236 shot188 - shot
      ingredient60 ingredient414 ingredient251 - ingredient
      cocktail1 - cocktail
      dispenser131 dispenser55 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot236)
  (ontable shot188)
  (dispenses dispenser131 ingredient60)
  (dispenses dispenser55 ingredient414)
  (dispenses dispenser229 ingredient251)
  (clean shaker370)
  (clean shot236)
  (clean shot188)
  (empty shaker370)
  (empty shot236)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient60)
  (cocktail-part2 cocktail1 ingredient414)
)
 (:goal
  (and
      (contains shot236 cocktail1)
)))
