(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot378 shot333 - shot
      ingredient418 ingredient199 ingredient84 ingredient18 - ingredient
      cocktail75 - cocktail
      dispenser179 dispenser388 dispenser488 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot378)
  (ontable shot333)
  (dispenses dispenser179 ingredient418)
  (dispenses dispenser388 ingredient199)
  (dispenses dispenser488 ingredient84)
  (dispenses dispenser103 ingredient18)
  (clean shaker272)
  (clean shot378)
  (clean shot333)
  (empty shaker272)
  (empty shot378)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient199)
  (cocktail-part2 cocktail75 ingredient84)
)
 (:goal
  (and
      (contains shot378 cocktail75)
)))
