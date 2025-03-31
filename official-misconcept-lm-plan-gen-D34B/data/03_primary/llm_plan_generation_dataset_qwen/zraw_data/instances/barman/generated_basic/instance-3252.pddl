(define (problem prob)
 (:domain barman)
 (:objects 
      shaker226 - shaker
      left right - hand
      shot52 shot355 shot83 - shot
      ingredient91 ingredient245 ingredient119 ingredient493 - ingredient
      cocktail176 - cocktail
      dispenser3 dispenser119 dispenser336 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker226)
  (ontable shot52)
  (ontable shot355)
  (ontable shot83)
  (dispenses dispenser3 ingredient91)
  (dispenses dispenser119 ingredient245)
  (dispenses dispenser336 ingredient119)
  (dispenses dispenser254 ingredient493)
  (clean shaker226)
  (clean shot52)
  (clean shot355)
  (clean shot83)
  (empty shaker226)
  (empty shot52)
  (empty shot355)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker226 l0)
  (shaker-level shaker226 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail176 ingredient493)
  (cocktail-part2 cocktail176 ingredient91)
)
 (:goal
  (and
      (contains shot52 cocktail176)
      (contains shot355 cocktail176)
)))
