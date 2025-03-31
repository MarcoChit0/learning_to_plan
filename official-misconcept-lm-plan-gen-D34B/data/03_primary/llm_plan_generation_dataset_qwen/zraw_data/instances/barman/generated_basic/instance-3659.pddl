(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot266 shot147 - shot
      ingredient273 ingredient163 ingredient448 ingredient95 - ingredient
      cocktail473 - cocktail
      dispenser160 dispenser83 dispenser17 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot266)
  (ontable shot147)
  (dispenses dispenser160 ingredient273)
  (dispenses dispenser83 ingredient163)
  (dispenses dispenser17 ingredient448)
  (dispenses dispenser429 ingredient95)
  (clean shaker219)
  (clean shot266)
  (clean shot147)
  (empty shaker219)
  (empty shot266)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient95)
  (cocktail-part2 cocktail473 ingredient163)
)
 (:goal
  (and
      (contains shot266 cocktail473)
)))
