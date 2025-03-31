(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot86 shot28 - shot
      ingredient102 ingredient358 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser27 dispenser143 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot86)
  (ontable shot28)
  (dispenses dispenser27 ingredient102)
  (dispenses dispenser143 ingredient358)
  (dispenses dispenser453 ingredient390)
  (clean shaker379)
  (clean shot86)
  (clean shot28)
  (empty shaker379)
  (empty shot86)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient102)
  (cocktail-part2 cocktail1 ingredient358)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))
