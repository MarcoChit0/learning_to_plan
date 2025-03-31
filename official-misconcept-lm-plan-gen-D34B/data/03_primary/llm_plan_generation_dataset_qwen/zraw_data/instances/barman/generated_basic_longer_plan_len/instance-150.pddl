(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot263 shot172 shot394 - shot
      ingredient139 ingredient279 ingredient246 - ingredient
      cocktail1 - cocktail
      dispenser268 dispenser173 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot263)
  (ontable shot172)
  (ontable shot394)
  (dispenses dispenser268 ingredient139)
  (dispenses dispenser173 ingredient279)
  (dispenses dispenser176 ingredient246)
  (clean shaker440)
  (clean shot263)
  (clean shot172)
  (clean shot394)
  (empty shaker440)
  (empty shot263)
  (empty shot172)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient279)
  (cocktail-part2 cocktail1 ingredient246)
)
 (:goal
  (and
      (contains shot263 cocktail1)
      (contains shot172 ingredient139)
)))
