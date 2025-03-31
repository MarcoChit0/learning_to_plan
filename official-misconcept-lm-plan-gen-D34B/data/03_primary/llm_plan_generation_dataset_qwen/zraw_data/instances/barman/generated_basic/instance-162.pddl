(define (problem prob)
 (:domain barman)
 (:objects 
      shaker69 - shaker
      left right - hand
      shot369 shot402 - shot
      ingredient137 ingredient191 ingredient162 - ingredient
      cocktail1 - cocktail
      dispenser104 dispenser72 dispenser53 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker69)
  (ontable shot369)
  (ontable shot402)
  (dispenses dispenser104 ingredient137)
  (dispenses dispenser72 ingredient191)
  (dispenses dispenser53 ingredient162)
  (clean shaker69)
  (clean shot369)
  (clean shot402)
  (empty shaker69)
  (empty shot369)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker69 l0)
  (shaker-level shaker69 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient191)
  (cocktail-part2 cocktail1 ingredient137)
)
 (:goal
  (and
      (contains shot369 cocktail1)
)))
