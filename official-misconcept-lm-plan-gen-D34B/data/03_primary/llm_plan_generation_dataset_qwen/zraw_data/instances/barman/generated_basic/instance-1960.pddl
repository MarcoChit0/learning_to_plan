(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot203 shot172 shot381 - shot
      ingredient269 ingredient109 ingredient461 ingredient472 - ingredient
      cocktail326 - cocktail
      dispenser392 dispenser271 dispenser22 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot203)
  (ontable shot172)
  (ontable shot381)
  (dispenses dispenser392 ingredient269)
  (dispenses dispenser271 ingredient109)
  (dispenses dispenser22 ingredient461)
  (dispenses dispenser64 ingredient472)
  (clean shaker295)
  (clean shot203)
  (clean shot172)
  (clean shot381)
  (empty shaker295)
  (empty shot203)
  (empty shot172)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient472)
  (cocktail-part2 cocktail326 ingredient461)
)
 (:goal
  (and
      (contains shot203 cocktail326)
      (contains shot172 ingredient269)
)))
