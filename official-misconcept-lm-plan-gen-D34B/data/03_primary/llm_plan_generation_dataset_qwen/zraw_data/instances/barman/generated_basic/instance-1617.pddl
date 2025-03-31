(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot48 shot308 - shot
      ingredient466 ingredient449 ingredient428 - ingredient
      cocktail245 - cocktail
      dispenser16 dispenser222 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot48)
  (ontable shot308)
  (dispenses dispenser16 ingredient466)
  (dispenses dispenser222 ingredient449)
  (dispenses dispenser68 ingredient428)
  (clean shaker90)
  (clean shot48)
  (clean shot308)
  (empty shaker90)
  (empty shot48)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient428)
  (cocktail-part2 cocktail245 ingredient466)
)
 (:goal
  (and
      (contains shot48 cocktail245)
)))
