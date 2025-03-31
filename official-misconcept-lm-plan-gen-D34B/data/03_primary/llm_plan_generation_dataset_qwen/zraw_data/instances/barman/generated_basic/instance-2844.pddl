(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot347 shot459 - shot
      ingredient247 ingredient260 ingredient144 ingredient33 - ingredient
      cocktail460 - cocktail
      dispenser167 dispenser168 dispenser325 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot347)
  (ontable shot459)
  (dispenses dispenser167 ingredient247)
  (dispenses dispenser168 ingredient260)
  (dispenses dispenser325 ingredient144)
  (dispenses dispenser305 ingredient33)
  (clean shaker329)
  (clean shot347)
  (clean shot459)
  (empty shaker329)
  (empty shot347)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient33)
  (cocktail-part2 cocktail460 ingredient260)
)
 (:goal
  (and
      (contains shot347 cocktail460)
)))
