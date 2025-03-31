(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot115 shot175 - shot
      ingredient345 ingredient469 ingredient36 - ingredient
      cocktail460 - cocktail
      dispenser443 dispenser178 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot115)
  (ontable shot175)
  (dispenses dispenser443 ingredient345)
  (dispenses dispenser178 ingredient469)
  (dispenses dispenser483 ingredient36)
  (clean shaker42)
  (clean shot115)
  (clean shot175)
  (empty shaker42)
  (empty shot115)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient469)
  (cocktail-part2 cocktail460 ingredient345)
)
 (:goal
  (and
      (contains shot115 cocktail460)
)))
