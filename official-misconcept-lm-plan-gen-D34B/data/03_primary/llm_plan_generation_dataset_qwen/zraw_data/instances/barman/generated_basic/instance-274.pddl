(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot394 shot400 - shot
      ingredient216 ingredient160 ingredient119 - ingredient
      cocktail1 - cocktail
      dispenser224 dispenser447 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot394)
  (ontable shot400)
  (dispenses dispenser224 ingredient216)
  (dispenses dispenser447 ingredient160)
  (dispenses dispenser112 ingredient119)
  (clean shaker120)
  (clean shot394)
  (clean shot400)
  (empty shaker120)
  (empty shot394)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient160)
  (cocktail-part2 cocktail1 ingredient216)
)
 (:goal
  (and
      (contains shot394 cocktail1)
)))
