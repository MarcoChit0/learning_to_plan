(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot351 shot206 - shot
      ingredient254 ingredient453 ingredient345 ingredient233 - ingredient
      cocktail164 - cocktail
      dispenser174 dispenser375 dispenser326 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot351)
  (ontable shot206)
  (dispenses dispenser174 ingredient254)
  (dispenses dispenser375 ingredient453)
  (dispenses dispenser326 ingredient345)
  (dispenses dispenser316 ingredient233)
  (clean shaker324)
  (clean shot351)
  (clean shot206)
  (empty shaker324)
  (empty shot351)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient453)
  (cocktail-part2 cocktail164 ingredient254)
)
 (:goal
  (and
      (contains shot351 cocktail164)
)))
