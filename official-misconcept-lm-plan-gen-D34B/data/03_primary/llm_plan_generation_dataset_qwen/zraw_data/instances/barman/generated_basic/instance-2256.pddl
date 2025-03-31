(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot276 shot165 shot74 - shot
      ingredient7 ingredient209 ingredient280 ingredient348 - ingredient
      cocktail396 - cocktail
      dispenser189 dispenser219 dispenser90 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot276)
  (ontable shot165)
  (ontable shot74)
  (dispenses dispenser189 ingredient7)
  (dispenses dispenser219 ingredient209)
  (dispenses dispenser90 ingredient280)
  (dispenses dispenser224 ingredient348)
  (clean shaker150)
  (clean shot276)
  (clean shot165)
  (clean shot74)
  (empty shaker150)
  (empty shot276)
  (empty shot165)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail396 ingredient209)
  (cocktail-part2 cocktail396 ingredient7)
)
 (:goal
  (and
      (contains shot276 cocktail396)
      (contains shot165 ingredient348)
)))
