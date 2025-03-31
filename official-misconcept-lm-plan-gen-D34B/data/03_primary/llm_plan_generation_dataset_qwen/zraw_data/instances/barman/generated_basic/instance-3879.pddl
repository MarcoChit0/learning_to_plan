(define (problem prob)
 (:domain barman)
 (:objects 
      shaker396 - shaker
      left right - hand
      shot100 shot365 - shot
      ingredient150 ingredient115 ingredient65 - ingredient
      cocktail229 - cocktail
      dispenser158 dispenser188 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker396)
  (ontable shot100)
  (ontable shot365)
  (dispenses dispenser158 ingredient150)
  (dispenses dispenser188 ingredient115)
  (dispenses dispenser165 ingredient65)
  (clean shaker396)
  (clean shot100)
  (clean shot365)
  (empty shaker396)
  (empty shot100)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker396 l0)
  (shaker-level shaker396 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient65)
  (cocktail-part2 cocktail229 ingredient115)
)
 (:goal
  (and
      (contains shot100 cocktail229)
)))
