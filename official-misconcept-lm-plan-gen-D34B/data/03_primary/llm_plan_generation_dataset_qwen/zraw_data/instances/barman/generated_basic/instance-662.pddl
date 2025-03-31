(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot158 shot439 - shot
      ingredient138 ingredient497 ingredient286 - ingredient
      cocktail1 - cocktail
      dispenser203 dispenser249 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot158)
  (ontable shot439)
  (dispenses dispenser203 ingredient138)
  (dispenses dispenser249 ingredient497)
  (dispenses dispenser322 ingredient286)
  (clean shaker458)
  (clean shot158)
  (clean shot439)
  (empty shaker458)
  (empty shot158)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient138)
)
 (:goal
  (and
      (contains shot158 cocktail1)
)))
