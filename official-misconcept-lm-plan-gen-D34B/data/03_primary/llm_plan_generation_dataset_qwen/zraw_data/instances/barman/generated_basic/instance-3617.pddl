(define (problem prob)
 (:domain barman)
 (:objects 
      shaker142 - shaker
      left right - hand
      shot245 shot215 shot230 - shot
      ingredient363 ingredient208 ingredient417 - ingredient
      cocktail176 - cocktail
      dispenser364 dispenser129 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker142)
  (ontable shot245)
  (ontable shot215)
  (ontable shot230)
  (dispenses dispenser364 ingredient363)
  (dispenses dispenser129 ingredient208)
  (dispenses dispenser223 ingredient417)
  (clean shaker142)
  (clean shot245)
  (clean shot215)
  (clean shot230)
  (empty shaker142)
  (empty shot245)
  (empty shot215)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker142 l0)
  (shaker-level shaker142 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail176 ingredient417)
  (cocktail-part2 cocktail176 ingredient208)
)
 (:goal
  (and
      (contains shot245 cocktail176)
      (contains shot215 cocktail176)
)))
