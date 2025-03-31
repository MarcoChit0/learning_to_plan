(define (problem prob)
 (:domain barman)
 (:objects 
      shaker396 - shaker
      left right - hand
      shot341 - shot
      ingredient57 ingredient228 ingredient223 - ingredient
      cocktail188 - cocktail
      dispenser134 dispenser262 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker396)
  (ontable shot341)
  (dispenses dispenser134 ingredient57)
  (dispenses dispenser262 ingredient228)
  (dispenses dispenser409 ingredient223)
  (clean shaker396)
  (clean shot341)
  (empty shaker396)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker396 l0)
  (shaker-level shaker396 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient223)
  (cocktail-part2 cocktail188 ingredient228)
)
 (:goal
  (and
      (contains shot341 cocktail188)
)))
