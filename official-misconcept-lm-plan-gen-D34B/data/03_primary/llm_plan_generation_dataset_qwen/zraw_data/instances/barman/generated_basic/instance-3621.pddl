(define (problem prob)
 (:domain barman)
 (:objects 
      shaker28 - shaker
      left right - hand
      shot24 shot183 shot312 - shot
      ingredient356 ingredient150 ingredient369 - ingredient
      cocktail250 - cocktail
      dispenser71 dispenser95 dispenser164 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker28)
  (ontable shot24)
  (ontable shot183)
  (ontable shot312)
  (dispenses dispenser71 ingredient356)
  (dispenses dispenser95 ingredient150)
  (dispenses dispenser164 ingredient369)
  (clean shaker28)
  (clean shot24)
  (clean shot183)
  (clean shot312)
  (empty shaker28)
  (empty shot24)
  (empty shot183)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker28 l0)
  (shaker-level shaker28 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient369)
  (cocktail-part2 cocktail250 ingredient356)
)
 (:goal
  (and
      (contains shot24 cocktail250)
      (contains shot183 ingredient369)
)))
