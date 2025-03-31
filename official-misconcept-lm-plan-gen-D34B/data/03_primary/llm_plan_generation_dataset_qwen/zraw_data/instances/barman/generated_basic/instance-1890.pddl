(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot115 shot53 - shot
      ingredient224 ingredient290 ingredient327 - ingredient
      cocktail354 - cocktail
      dispenser38 dispenser316 dispenser199 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot115)
  (ontable shot53)
  (dispenses dispenser38 ingredient224)
  (dispenses dispenser316 ingredient290)
  (dispenses dispenser199 ingredient327)
  (clean shaker446)
  (clean shot115)
  (clean shot53)
  (empty shaker446)
  (empty shot115)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail354 ingredient327)
  (cocktail-part2 cocktail354 ingredient224)
)
 (:goal
  (and
      (contains shot115 cocktail354)
)))
