(define (problem prob)
 (:domain barman)
 (:objects 
      shaker267 - shaker
      left right - hand
      shot474 shot341 shot263 - shot
      ingredient140 ingredient478 - ingredient
      cocktail206 - cocktail
      dispenser424 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker267)
  (ontable shot474)
  (ontable shot341)
  (ontable shot263)
  (dispenses dispenser424 ingredient140)
  (dispenses dispenser234 ingredient478)
  (clean shaker267)
  (clean shot474)
  (clean shot341)
  (clean shot263)
  (empty shaker267)
  (empty shot474)
  (empty shot341)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker267 l0)
  (shaker-level shaker267 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient478)
  (cocktail-part2 cocktail206 ingredient140)
)
 (:goal
  (and
      (contains shot474 cocktail206)
      (contains shot341 ingredient140)
)))
