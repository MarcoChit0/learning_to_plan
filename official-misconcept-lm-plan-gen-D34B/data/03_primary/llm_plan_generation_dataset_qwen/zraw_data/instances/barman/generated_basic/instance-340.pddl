(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot359 shot413 - shot
      ingredient54 ingredient88 ingredient152 ingredient155 - ingredient
      cocktail1 - cocktail
      dispenser56 dispenser448 dispenser80 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot359)
  (ontable shot413)
  (dispenses dispenser56 ingredient54)
  (dispenses dispenser448 ingredient88)
  (dispenses dispenser80 ingredient152)
  (dispenses dispenser492 ingredient155)
  (clean shaker74)
  (clean shot359)
  (clean shot413)
  (empty shaker74)
  (empty shot359)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient54)
  (cocktail-part2 cocktail1 ingredient88)
)
 (:goal
  (and
      (contains shot359 cocktail1)
)))
