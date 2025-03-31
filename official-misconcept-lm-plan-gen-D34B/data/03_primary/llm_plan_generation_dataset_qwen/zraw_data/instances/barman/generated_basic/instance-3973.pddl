(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot64 shot212 - shot
      ingredient250 ingredient145 ingredient77 ingredient45 - ingredient
      cocktail349 - cocktail
      dispenser388 dispenser91 dispenser494 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot64)
  (ontable shot212)
  (dispenses dispenser388 ingredient250)
  (dispenses dispenser91 ingredient145)
  (dispenses dispenser494 ingredient77)
  (dispenses dispenser102 ingredient45)
  (clean shaker43)
  (clean shot64)
  (clean shot212)
  (empty shaker43)
  (empty shot64)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail349 ingredient45)
  (cocktail-part2 cocktail349 ingredient145)
)
 (:goal
  (and
      (contains shot64 cocktail349)
)))
