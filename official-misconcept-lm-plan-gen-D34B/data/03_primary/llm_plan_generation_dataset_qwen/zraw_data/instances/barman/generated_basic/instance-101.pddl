(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot440 shot390 shot91 - shot
      ingredient13 ingredient77 ingredient197 - ingredient
      cocktail1 - cocktail
      dispenser46 dispenser202 dispenser104 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot440)
  (ontable shot390)
  (ontable shot91)
  (dispenses dispenser46 ingredient13)
  (dispenses dispenser202 ingredient77)
  (dispenses dispenser104 ingredient197)
  (clean shaker388)
  (clean shot440)
  (clean shot390)
  (clean shot91)
  (empty shaker388)
  (empty shot440)
  (empty shot390)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient13)
  (cocktail-part2 cocktail1 ingredient77)
)
 (:goal
  (and
      (contains shot440 cocktail1)
      (contains shot390 ingredient77)
)))
