(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot229 shot434 shot179 - shot
      ingredient390 ingredient385 ingredient163 ingredient489 - ingredient
      cocktail110 - cocktail
      dispenser219 dispenser404 dispenser52 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot229)
  (ontable shot434)
  (ontable shot179)
  (dispenses dispenser219 ingredient390)
  (dispenses dispenser404 ingredient385)
  (dispenses dispenser52 ingredient163)
  (dispenses dispenser383 ingredient489)
  (clean shaker185)
  (clean shot229)
  (clean shot434)
  (clean shot179)
  (empty shaker185)
  (empty shot229)
  (empty shot434)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient489)
  (cocktail-part2 cocktail110 ingredient385)
)
 (:goal
  (and
      (contains shot229 cocktail110)
      (contains shot434 cocktail110)
)))
