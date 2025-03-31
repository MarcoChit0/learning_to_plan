(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot444 shot12 - shot
      ingredient335 ingredient251 ingredient303 ingredient0 - ingredient
      cocktail1 - cocktail
      dispenser464 dispenser400 dispenser312 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot444)
  (ontable shot12)
  (dispenses dispenser464 ingredient335)
  (dispenses dispenser400 ingredient251)
  (dispenses dispenser312 ingredient303)
  (dispenses dispenser433 ingredient0)
  (clean shaker308)
  (clean shot444)
  (clean shot12)
  (empty shaker308)
  (empty shot444)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient335)
  (cocktail-part2 cocktail1 ingredient0)
)
 (:goal
  (and
      (contains shot444 cocktail1)
)))
