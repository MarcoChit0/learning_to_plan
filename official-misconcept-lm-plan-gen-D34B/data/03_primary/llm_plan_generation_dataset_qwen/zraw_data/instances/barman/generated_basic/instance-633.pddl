(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot392 shot481 - shot
      ingredient129 ingredient21 ingredient362 ingredient289 - ingredient
      cocktail1 - cocktail
      dispenser194 dispenser325 dispenser267 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot392)
  (ontable shot481)
  (dispenses dispenser194 ingredient129)
  (dispenses dispenser325 ingredient21)
  (dispenses dispenser267 ingredient362)
  (dispenses dispenser114 ingredient289)
  (clean shaker174)
  (clean shot392)
  (clean shot481)
  (empty shaker174)
  (empty shot392)
  (empty shot481)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient362)
)
 (:goal
  (and
      (contains shot392 cocktail1)
)))
