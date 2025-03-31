(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot56 shot490 - shot
      ingredient154 ingredient266 ingredient327 ingredient139 - ingredient
      cocktail377 - cocktail
      dispenser482 dispenser40 dispenser406 dispenser318 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot56)
  (ontable shot490)
  (dispenses dispenser482 ingredient154)
  (dispenses dispenser40 ingredient266)
  (dispenses dispenser406 ingredient327)
  (dispenses dispenser318 ingredient139)
  (clean shaker224)
  (clean shot56)
  (clean shot490)
  (empty shaker224)
  (empty shot56)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail377 ingredient266)
  (cocktail-part2 cocktail377 ingredient154)
)
 (:goal
  (and
      (contains shot56 cocktail377)
)))
