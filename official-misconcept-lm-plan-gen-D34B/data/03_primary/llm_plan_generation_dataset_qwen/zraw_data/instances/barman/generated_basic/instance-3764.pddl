(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot325 shot186 shot189 - shot
      ingredient48 ingredient161 - ingredient
      cocktail104 - cocktail
      dispenser138 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot325)
  (ontable shot186)
  (ontable shot189)
  (dispenses dispenser138 ingredient48)
  (dispenses dispenser275 ingredient161)
  (clean shaker380)
  (clean shot325)
  (clean shot186)
  (clean shot189)
  (empty shaker380)
  (empty shot325)
  (empty shot186)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient48)
  (cocktail-part2 cocktail104 ingredient161)
)
 (:goal
  (and
      (contains shot325 cocktail104)
      (contains shot186 ingredient161)
)))
