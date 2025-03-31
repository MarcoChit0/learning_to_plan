(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot26 - shot
      ingredient85 ingredient109 ingredient319 ingredient38 - ingredient
      cocktail1 - cocktail
      dispenser365 dispenser265 dispenser154 dispenser488 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot26)
  (dispenses dispenser365 ingredient85)
  (dispenses dispenser265 ingredient109)
  (dispenses dispenser154 ingredient319)
  (dispenses dispenser488 ingredient38)
  (clean shaker210)
  (clean shot26)
  (empty shaker210)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient85)
  (cocktail-part2 cocktail1 ingredient319)
)
 (:goal
  (and
      (contains shot26 cocktail1)
)))
