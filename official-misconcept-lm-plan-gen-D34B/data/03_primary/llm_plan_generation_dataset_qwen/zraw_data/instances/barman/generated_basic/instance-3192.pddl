(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot241 shot114 - shot
      ingredient140 ingredient206 ingredient279 - ingredient
      cocktail311 - cocktail
      dispenser158 dispenser254 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot241)
  (ontable shot114)
  (dispenses dispenser158 ingredient140)
  (dispenses dispenser254 ingredient206)
  (dispenses dispenser464 ingredient279)
  (clean shaker374)
  (clean shot241)
  (clean shot114)
  (empty shaker374)
  (empty shot241)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient140)
  (cocktail-part2 cocktail311 ingredient206)
)
 (:goal
  (and
      (contains shot241 cocktail311)
)))
