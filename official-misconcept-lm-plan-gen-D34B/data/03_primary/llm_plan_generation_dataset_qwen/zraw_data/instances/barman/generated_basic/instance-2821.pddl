(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot330 shot12 shot92 - shot
      ingredient234 ingredient126 ingredient84 ingredient226 - ingredient
      cocktail85 - cocktail
      dispenser204 dispenser159 dispenser323 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot330)
  (ontable shot12)
  (ontable shot92)
  (dispenses dispenser204 ingredient234)
  (dispenses dispenser159 ingredient126)
  (dispenses dispenser323 ingredient84)
  (dispenses dispenser403 ingredient226)
  (clean shaker99)
  (clean shot330)
  (clean shot12)
  (clean shot92)
  (empty shaker99)
  (empty shot330)
  (empty shot12)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient126)
  (cocktail-part2 cocktail85 ingredient84)
)
 (:goal
  (and
      (contains shot330 cocktail85)
      (contains shot12 cocktail85)
)))
