(define (problem prob)
 (:domain barman)
 (:objects 
      shaker240 - shaker
      left right - hand
      shot155 shot99 shot344 - shot
      ingredient486 ingredient219 ingredient480 - ingredient
      cocktail360 - cocktail
      dispenser5 dispenser482 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker240)
  (ontable shot155)
  (ontable shot99)
  (ontable shot344)
  (dispenses dispenser5 ingredient486)
  (dispenses dispenser482 ingredient219)
  (dispenses dispenser495 ingredient480)
  (clean shaker240)
  (clean shot155)
  (clean shot99)
  (clean shot344)
  (empty shaker240)
  (empty shot155)
  (empty shot99)
  (empty shot344)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker240 l0)
  (shaker-level shaker240 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient480)
  (cocktail-part2 cocktail360 ingredient486)
)
 (:goal
  (and
      (contains shot155 cocktail360)
      (contains shot99 cocktail360)
)))
