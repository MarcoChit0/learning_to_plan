(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot316 - shot
      ingredient142 ingredient395 - ingredient
      cocktail225 - cocktail
      dispenser177 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot316)
  (dispenses dispenser177 ingredient142)
  (dispenses dispenser224 ingredient395)
  (clean shaker433)
  (clean shot316)
  (empty shaker433)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient395)
  (cocktail-part2 cocktail225 ingredient142)
)
 (:goal
  (and
      (contains shot316 cocktail225)
)))
