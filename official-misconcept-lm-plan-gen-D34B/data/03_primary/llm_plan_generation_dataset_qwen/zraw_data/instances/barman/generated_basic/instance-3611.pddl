(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot159 shot173 shot192 - shot
      ingredient7 ingredient390 ingredient387 - ingredient
      cocktail361 - cocktail
      dispenser366 dispenser334 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot159)
  (ontable shot173)
  (ontable shot192)
  (dispenses dispenser366 ingredient7)
  (dispenses dispenser334 ingredient390)
  (dispenses dispenser251 ingredient387)
  (clean shaker200)
  (clean shot159)
  (clean shot173)
  (clean shot192)
  (empty shaker200)
  (empty shot159)
  (empty shot173)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail361 ingredient390)
  (cocktail-part2 cocktail361 ingredient7)
)
 (:goal
  (and
      (contains shot159 cocktail361)
      (contains shot173 ingredient7)
)))
