(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot291 shot220 - shot
      ingredient414 ingredient498 ingredient5 ingredient143 - ingredient
      cocktail12 - cocktail
      dispenser204 dispenser182 dispenser254 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot291)
  (ontable shot220)
  (dispenses dispenser204 ingredient414)
  (dispenses dispenser182 ingredient498)
  (dispenses dispenser254 ingredient5)
  (dispenses dispenser247 ingredient143)
  (clean shaker442)
  (clean shot291)
  (clean shot220)
  (empty shaker442)
  (empty shot291)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient143)
  (cocktail-part2 cocktail12 ingredient5)
)
 (:goal
  (and
      (contains shot291 cocktail12)
)))
