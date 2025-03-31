(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot66 shot27 shot341 - shot
      ingredient29 ingredient9 ingredient468 ingredient33 - ingredient
      cocktail1 - cocktail
      dispenser46 dispenser5 dispenser331 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot66)
  (ontable shot27)
  (ontable shot341)
  (dispenses dispenser46 ingredient29)
  (dispenses dispenser5 ingredient9)
  (dispenses dispenser331 ingredient468)
  (dispenses dispenser196 ingredient33)
  (clean shaker77)
  (clean shot66)
  (clean shot27)
  (clean shot341)
  (empty shaker77)
  (empty shot66)
  (empty shot27)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient29)
  (cocktail-part2 cocktail1 ingredient9)
)
 (:goal
  (and
      (contains shot66 cocktail1)
      (contains shot27 ingredient9)
)))
