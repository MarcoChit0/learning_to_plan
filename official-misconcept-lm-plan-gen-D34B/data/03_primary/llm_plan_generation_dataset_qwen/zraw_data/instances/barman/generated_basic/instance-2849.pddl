(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot324 shot5 - shot
      ingredient185 ingredient224 ingredient156 ingredient307 - ingredient
      cocktail260 - cocktail
      dispenser308 dispenser63 dispenser299 dispenser99 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot324)
  (ontable shot5)
  (dispenses dispenser308 ingredient185)
  (dispenses dispenser63 ingredient224)
  (dispenses dispenser299 ingredient156)
  (dispenses dispenser99 ingredient307)
  (clean shaker77)
  (clean shot324)
  (clean shot5)
  (empty shaker77)
  (empty shot324)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient307)
  (cocktail-part2 cocktail260 ingredient185)
)
 (:goal
  (and
      (contains shot324 cocktail260)
)))
