(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot472 shot99 shot160 - shot
      ingredient76 ingredient255 - ingredient
      cocktail187 - cocktail
      dispenser138 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot472)
  (ontable shot99)
  (ontable shot160)
  (dispenses dispenser138 ingredient76)
  (dispenses dispenser331 ingredient255)
  (clean shaker324)
  (clean shot472)
  (clean shot99)
  (clean shot160)
  (empty shaker324)
  (empty shot472)
  (empty shot99)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail187 ingredient255)
  (cocktail-part2 cocktail187 ingredient76)
)
 (:goal
  (and
      (contains shot472 cocktail187)
      (contains shot99 cocktail187)
)))
