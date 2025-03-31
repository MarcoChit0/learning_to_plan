(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot152 shot333 shot289 - shot
      ingredient52 ingredient208 ingredient427 ingredient135 - ingredient
      cocktail75 - cocktail
      dispenser471 dispenser477 dispenser59 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot152)
  (ontable shot333)
  (ontable shot289)
  (dispenses dispenser471 ingredient52)
  (dispenses dispenser477 ingredient208)
  (dispenses dispenser59 ingredient427)
  (dispenses dispenser328 ingredient135)
  (clean shaker441)
  (clean shot152)
  (clean shot333)
  (clean shot289)
  (empty shaker441)
  (empty shot152)
  (empty shot333)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient427)
  (cocktail-part2 cocktail75 ingredient208)
)
 (:goal
  (and
      (contains shot152 cocktail75)
      (contains shot333 ingredient208)
)))
