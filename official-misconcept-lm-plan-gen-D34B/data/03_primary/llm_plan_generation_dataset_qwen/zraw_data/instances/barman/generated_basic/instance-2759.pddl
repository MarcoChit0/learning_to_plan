(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot295 shot154 - shot
      ingredient131 ingredient196 ingredient220 - ingredient
      cocktail422 - cocktail
      dispenser294 dispenser310 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot295)
  (ontable shot154)
  (dispenses dispenser294 ingredient131)
  (dispenses dispenser310 ingredient196)
  (dispenses dispenser425 ingredient220)
  (clean shaker260)
  (clean shot295)
  (clean shot154)
  (empty shaker260)
  (empty shot295)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient196)
  (cocktail-part2 cocktail422 ingredient220)
)
 (:goal
  (and
      (contains shot295 cocktail422)
)))
