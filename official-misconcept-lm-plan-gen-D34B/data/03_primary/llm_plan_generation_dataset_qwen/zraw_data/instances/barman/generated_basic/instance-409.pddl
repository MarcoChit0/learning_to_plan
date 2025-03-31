(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot420 shot413 - shot
      ingredient310 ingredient258 - ingredient
      cocktail1 - cocktail
      dispenser146 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot420)
  (ontable shot413)
  (dispenses dispenser146 ingredient310)
  (dispenses dispenser398 ingredient258)
  (clean shaker149)
  (clean shot420)
  (clean shot413)
  (empty shaker149)
  (empty shot420)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient310)
  (cocktail-part2 cocktail1 ingredient258)
)
 (:goal
  (and
      (contains shot420 cocktail1)
)))
