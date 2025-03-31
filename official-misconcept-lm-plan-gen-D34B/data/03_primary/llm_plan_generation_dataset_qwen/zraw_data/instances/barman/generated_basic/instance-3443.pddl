(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot303 shot184 - shot
      ingredient230 ingredient196 ingredient60 ingredient136 - ingredient
      cocktail345 - cocktail
      dispenser453 dispenser134 dispenser446 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot303)
  (ontable shot184)
  (dispenses dispenser453 ingredient230)
  (dispenses dispenser134 ingredient196)
  (dispenses dispenser446 ingredient60)
  (dispenses dispenser171 ingredient136)
  (clean shaker216)
  (clean shot303)
  (clean shot184)
  (empty shaker216)
  (empty shot303)
  (empty shot184)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail345 ingredient136)
  (cocktail-part2 cocktail345 ingredient230)
)
 (:goal
  (and
      (contains shot303 cocktail345)
)))
