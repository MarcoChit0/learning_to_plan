(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot94 shot480 - shot
      ingredient203 ingredient269 ingredient240 - ingredient
      cocktail25 - cocktail
      dispenser464 dispenser451 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot94)
  (ontable shot480)
  (dispenses dispenser464 ingredient203)
  (dispenses dispenser451 ingredient269)
  (dispenses dispenser165 ingredient240)
  (clean shaker383)
  (clean shot94)
  (clean shot480)
  (empty shaker383)
  (empty shot94)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail25 ingredient269)
  (cocktail-part2 cocktail25 ingredient240)
)
 (:goal
  (and
      (contains shot94 cocktail25)
)))
