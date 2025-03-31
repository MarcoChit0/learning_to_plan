(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot154 shot261 - shot
      ingredient418 ingredient412 ingredient416 - ingredient
      cocktail149 - cocktail
      dispenser102 dispenser314 dispenser27 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot154)
  (ontable shot261)
  (dispenses dispenser102 ingredient418)
  (dispenses dispenser314 ingredient412)
  (dispenses dispenser27 ingredient416)
  (clean shaker433)
  (clean shot154)
  (clean shot261)
  (empty shaker433)
  (empty shot154)
  (empty shot261)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail149 ingredient416)
  (cocktail-part2 cocktail149 ingredient418)
)
 (:goal
  (and
      (contains shot154 cocktail149)
)))
