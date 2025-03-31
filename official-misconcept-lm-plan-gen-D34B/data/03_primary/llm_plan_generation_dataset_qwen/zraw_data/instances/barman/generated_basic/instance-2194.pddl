(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot203 shot326 shot345 - shot
      ingredient125 ingredient498 - ingredient
      cocktail180 - cocktail
      dispenser50 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot203)
  (ontable shot326)
  (ontable shot345)
  (dispenses dispenser50 ingredient125)
  (dispenses dispenser246 ingredient498)
  (clean shaker425)
  (clean shot203)
  (clean shot326)
  (clean shot345)
  (empty shaker425)
  (empty shot203)
  (empty shot326)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail180 ingredient498)
  (cocktail-part2 cocktail180 ingredient125)
)
 (:goal
  (and
      (contains shot203 cocktail180)
      (contains shot326 ingredient498)
)))
