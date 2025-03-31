(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot188 shot446 - shot
      ingredient163 ingredient415 ingredient290 ingredient56 - ingredient
      cocktail420 - cocktail
      dispenser143 dispenser152 dispenser59 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot188)
  (ontable shot446)
  (dispenses dispenser143 ingredient163)
  (dispenses dispenser152 ingredient415)
  (dispenses dispenser59 ingredient290)
  (dispenses dispenser475 ingredient56)
  (clean shaker318)
  (clean shot188)
  (clean shot446)
  (empty shaker318)
  (empty shot188)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail420 ingredient163)
  (cocktail-part2 cocktail420 ingredient290)
)
 (:goal
  (and
      (contains shot188 cocktail420)
)))
