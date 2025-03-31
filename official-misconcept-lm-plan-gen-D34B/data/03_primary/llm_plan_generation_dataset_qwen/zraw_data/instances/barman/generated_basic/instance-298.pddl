(define (problem prob)
 (:domain barman)
 (:objects 
      shaker394 - shaker
      left right - hand
      shot393 shot300 shot105 - shot
      ingredient250 ingredient146 ingredient121 - ingredient
      cocktail1 - cocktail
      dispenser258 dispenser264 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker394)
  (ontable shot393)
  (ontable shot300)
  (ontable shot105)
  (dispenses dispenser258 ingredient250)
  (dispenses dispenser264 ingredient146)
  (dispenses dispenser271 ingredient121)
  (clean shaker394)
  (clean shot393)
  (clean shot300)
  (clean shot105)
  (empty shaker394)
  (empty shot393)
  (empty shot300)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker394 l0)
  (shaker-level shaker394 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient121)
  (cocktail-part2 cocktail1 ingredient250)
)
 (:goal
  (and
      (contains shot393 cocktail1)
      (contains shot300 cocktail1)
)))
