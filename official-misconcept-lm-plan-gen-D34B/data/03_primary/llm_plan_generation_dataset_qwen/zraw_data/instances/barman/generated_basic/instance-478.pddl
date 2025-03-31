(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot278 shot186 - shot
      ingredient31 ingredient284 ingredient267 ingredient234 - ingredient
      cocktail1 - cocktail
      dispenser52 dispenser314 dispenser407 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot278)
  (ontable shot186)
  (dispenses dispenser52 ingredient31)
  (dispenses dispenser314 ingredient284)
  (dispenses dispenser407 ingredient267)
  (dispenses dispenser308 ingredient234)
  (clean shaker125)
  (clean shot278)
  (clean shot186)
  (empty shaker125)
  (empty shot278)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient267)
  (cocktail-part2 cocktail1 ingredient284)
)
 (:goal
  (and
      (contains shot278 cocktail1)
)))
