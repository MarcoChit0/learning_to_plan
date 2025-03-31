(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot472 shot168 - shot
      ingredient269 ingredient365 ingredient315 - ingredient
      cocktail74 - cocktail
      dispenser240 dispenser94 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot472)
  (ontable shot168)
  (dispenses dispenser240 ingredient269)
  (dispenses dispenser94 ingredient365)
  (dispenses dispenser13 ingredient315)
  (clean shaker199)
  (clean shot472)
  (clean shot168)
  (empty shaker199)
  (empty shot472)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient365)
  (cocktail-part2 cocktail74 ingredient269)
)
 (:goal
  (and
      (contains shot472 cocktail74)
)))
