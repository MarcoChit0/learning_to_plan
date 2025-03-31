(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot490 shot290 - shot
      ingredient358 ingredient220 ingredient485 - ingredient
      cocktail3 - cocktail
      dispenser209 dispenser371 dispenser111 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot490)
  (ontable shot290)
  (dispenses dispenser209 ingredient358)
  (dispenses dispenser371 ingredient220)
  (dispenses dispenser111 ingredient485)
  (clean shaker296)
  (clean shot490)
  (clean shot290)
  (empty shaker296)
  (empty shot490)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient358)
  (cocktail-part2 cocktail3 ingredient220)
)
 (:goal
  (and
      (contains shot490 cocktail3)
)))
